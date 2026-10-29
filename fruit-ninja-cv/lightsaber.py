"""Draw a lightsaber at the tracked pen point.

The saber stays at the pen: the tip of the blade is the point that cuts the
fruit (the pen tip). The hand holding the pen decides the direction; the blade
length settles slowly (5 % per frame) so that the hilt ends up on the hand and
then stays practically constant -- jitter of the hand detection averages out
instead of stretching or squashing the saber. Against jitter the direction
is smoothed, a sudden large change is only taken over once it lasts a few
frames, the tilt is limited to MAX_TILT from vertical (skin above the tip, e.g.
the face, is never the hand), and without a hand the saber swings back to
vertical gradually. Drawing only -- the cutting still uses the pen point alone.
"""

import math
from collections import namedtuple

import cv2
import numpy as np

import config

Saber = namedtuple("Saber", "tip hilt_top hilt_bottom")


def saber_at(point, blade_length=None, hilt_length=None):
    """Saber geometry hanging down from ``point`` (the blade tip), or None."""
    if point is None:
        return None
    if blade_length is None:
        blade_length = round(config.SABER_LENGTH * config.SCALE)
    if hilt_length is None:
        hilt_length = round(config.SABER_HILT_LENGTH * config.SCALE)
    x, y = int(point[0]), int(point[1])
    return Saber((x, y), (x, y + blade_length), (x, y + blade_length + hilt_length))


def find_hand(skin, tip, radius, exclude=None, min_area=None, hand_size=None):
    """Where the hand holding the pen is, or None.

    The nearest skin blob (by its closest pixel) within ``radius`` of ``tip``;
    blobs centred in ``exclude`` (the face box x, y, w, h) are ignored. The
    result is the centre of the part of that blob next to the pen (within
    ``hand_size`` of its closest pixel) -- a hand touching a bare arm or a
    skin-coloured object must not drift into the middle of it all.
    """
    if skin is None:
        return None
    if min_area is None:
        min_area = 400 * config.SCALE ** 2
    if hand_size is None:
        hand_size = 80 * config.SCALE
    h, w = skin.shape[:2]
    x0, y0 = max(int(tip[0] - radius), 0), max(int(tip[1] - radius), 0)
    x1, y1 = min(int(tip[0] + radius) + 1, w), min(int(tip[1] + radius) + 1, h)
    if x0 >= x1 or y0 >= y1:
        return None
    window = cv2.threshold(skin[y0:y1, x0:x1], 127, 255, cv2.THRESH_BINARY)[1]
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(window, connectivity=8)
    if n <= 1:
        return None
    # distances only for the skin pixels (far fewer than the whole window)
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs]
    xs = xs.astype(np.float32) + x0
    ys = ys.astype(np.float32) + y0
    dist = np.hypot(xs - tip[0], ys - tip[1])
    nearest = np.full(n, np.inf, np.float32)
    np.minimum.at(nearest, lab, dist)                # closest pixel of every blob
    best, best_dist = None, None
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area or nearest[i] > radius:
            continue
        if exclude is not None:
            cx, cy = centroids[i][0] + x0, centroids[i][1] + y0
            ex, ey, ew, eh = exclude
            if ex - 0.3 * ew <= cx <= ex + 1.3 * ew and ey - 0.3 * eh <= cy <= ey + 1.3 * eh:
                continue
        if best_dist is None or nearest[i] < best_dist:
            best, best_dist = i, nearest[i]
    if best is None:
        return None
    part = (lab == best) & (dist <= best_dist + hand_size)
    return float(xs[part].mean()), float(ys[part].mean())


def saber_toward(tip, direction, blade_length=None, hilt_length=None):
    """Saber of fixed length from ``tip`` along ``direction`` (unit vector, blade then hilt)."""
    if blade_length is None:
        blade_length = config.SABER_LENGTH * config.SCALE
    if hilt_length is None:
        hilt_length = config.SABER_HILT_LENGTH * config.SCALE
    dx, dy = direction
    as_int = lambda q: (int(round(q[0])), int(round(q[1])))
    top = (tip[0] + dx * blade_length, tip[1] + dy * blade_length)
    bottom = (top[0] + dx * hilt_length, top[1] + dy * hilt_length)
    return Saber((int(tip[0]), int(tip[1])), as_int(top), as_int(bottom))


MAX_TILT = math.radians(75)      # largest tilt from straight down


class SaberTracker:
    """Saber direction frame by frame: towards the hand, steady, fixed length."""

    def __init__(self, smoothing=0.6, blade_length=None, hilt_length=None, radius=None,
                 confirm_frames=4, jump_angle=45, keep_frames=4, return_rate=0.15,
                 adapt_rate=0.05, min_blade=None, max_blade=None):
        self.smoothing = smoothing
        self.blade_length = blade_length             # start value (config if None)
        self.adapt_rate = adapt_rate                 # how fast the length follows the hand
        self.min_blade = min_blade
        self.max_blade = max_blade
        self.hilt_length = hilt_length
        self.radius = radius
        self.confirm_frames = confirm_frames         # frames a big change must last
        self.jump = math.radians(jump_angle)         # what counts as a big change
        self.keep_frames = keep_frames               # hold the direction while the hand is missed
        self.return_rate = return_rate               # then ease back to vertical per frame
        self._reset()

    def _reset(self):
        self.angle = 0.0                             # 0 = straight down, + = hilt to the right
        self.length = None                           # current blade length (settles on the hand)
        self._pending = 0
        self._missing = 0

    def _target(self, tip, skin, face):
        radius = self.radius if self.radius is not None else config.SABER_HAND_RADIUS * config.SCALE
        hand = find_hand(skin, tip, radius, face)
        if hand is None:
            return None
        dx, dy = hand[0] - tip[0], hand[1] - tip[1]
        if dy <= 0:                                  # hand above the tip: not the hand
            return None
        angle = math.atan2(dx, dy)
        return (angle, math.hypot(dx, dy)) if abs(angle) <= MAX_TILT else None

    def _adapt_length(self, hand_dist, hilt):
        """Let the blade grow/shrink slowly so the hilt centre ends up on the hand."""
        s = config.SCALE
        lo = self.min_blade if self.min_blade is not None else 60 * s
        hi = self.max_blade if self.max_blade is not None else 320 * s
        wanted = min(max(hand_dist - hilt / 2, lo), hi)
        self.length += (wanted - self.length) * self.adapt_rate

    def update(self, tip, skin, face=None):
        if tip is None:
            self._reset()
            return None
        hilt = self.hilt_length if self.hilt_length is not None else \
            config.SABER_HILT_LENGTH * config.SCALE
        if self.length is None:
            self.length = self.blade_length if self.blade_length is not None else \
                config.SABER_LENGTH * config.SCALE
        found = self._target(tip, skin, face)
        target = found[0] if found is not None else None
        if target is None:
            self._pending = 0
            self._missing += 1
            if self._missing > self.keep_frames:     # no hand: swing back slowly
                self.angle -= self.angle * self.return_rate
        else:
            self._missing = 0
            if abs(target - self.angle) > self.jump:
                self._pending += 1                   # big change: only if it lasts
                if self._pending >= self.confirm_frames:
                    self.angle, self._pending = target, 0
                    self._adapt_length(found[1], hilt)
            else:
                self._pending = 0
                k = self.smoothing
                self.angle = k * self.angle + (1 - k) * target
                self._adapt_length(found[1], hilt)   # only for accepted, not outlier frames
        direction = (math.sin(self.angle), math.cos(self.angle))
        return saber_toward(tip, direction, self.length, hilt)


def _add_glow(frame, a, b, color, thickness, sigma):
    """Add a blurred, coloured line from ``a`` to ``b`` (additive light) in place."""
    h, w = frame.shape[:2]
    margin = int(thickness + 3 * sigma)
    x0 = max(min(a[0], b[0]) - margin, 0)
    y0 = max(min(a[1], b[1]) - margin, 0)
    x1 = min(max(a[0], b[0]) + margin, w)
    y1 = min(max(a[1], b[1]) + margin, h)
    if x0 >= x1 or y0 >= y1:
        return
    # the glow is soft anyway: draw and blur at half resolution, then scale up
    hw, hh = max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)
    canvas = np.zeros((hh, hw, 3), np.uint8)
    cv2.line(canvas, ((a[0] - x0) // 2, (a[1] - y0) // 2), ((b[0] - x0) // 2, (b[1] - y0) // 2),
             color, max(1, thickness // 2), cv2.LINE_AA)
    canvas = cv2.GaussianBlur(canvas, (0, 0), max(0.5, sigma / 2))
    canvas = cv2.resize(canvas, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    roi = frame[y0:y1, x0:x1]
    cv2.add(roi, canvas, dst=roi)


def draw_saber(frame, saber, color):
    """Draw blade (thick glow + white core) and hilt into ``frame`` (in place)."""
    if saber is None:
        return frame
    s = config.SCALE
    tip, emitter, pommel = saber
    # blade: wide soft glow, tighter bright glow, thick white-hot core (round tip)
    _add_glow(frame, tip, emitter, color, max(4, int(44 * s)), max(1.0, 14 * s))
    _add_glow(frame, tip, emitter, color, max(3, int(24 * s)), max(1.0, 5 * s))
    cv2.line(frame, tip, emitter, (255, 255, 255), max(3, int(12 * s)), cv2.LINE_AA)

    # hilt along the saber axis: dark metal grip with ridges, bright emitter ring
    # at the blade end, lighter pommel cap at the other end
    width = max(6, int(18 * s))
    length = math.dist(emitter, pommel) or 1.0
    d = ((pommel[0] - emitter[0]) / length, (pommel[1] - emitter[1]) / length)
    perp = (-d[1], d[0])

    def across(t, half, color, thickness):
        cx, cy = emitter[0] + d[0] * length * t, emitter[1] + d[1] * length * t
        cv2.line(frame, (int(round(cx - perp[0] * half)), int(round(cy - perp[1] * half))),
                 (int(round(cx + perp[0] * half)), int(round(cy + perp[1] * half))),
                 color, thickness, cv2.LINE_AA)

    cv2.line(frame, emitter, pommel, (60, 60, 65), width)
    for t in (0.35, 0.5, 0.65, 0.8):
        across(t, width / 2, (25, 25, 30), max(1, int(3 * s)))
    across(0.04, width / 2 + max(1, int(3 * s)), (190, 190, 195), max(2, int(6 * s)))
    across(0.96, width / 2, (120, 120, 125), max(2, int(5 * s)))
    return frame
