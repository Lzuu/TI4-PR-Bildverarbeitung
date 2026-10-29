"""Draw a lightsaber at the tracked pen point.

The tip of the blade is the point that cuts the fruit (the pen tip). The hilt
sits on the hand holding the pen, so the blade runs from the tip to the hand
and tilts with it. The hand is the nearest sizeable skin-coloured blob around
the tip (the face is excluded). Its offset to the tip is smoothed, and a hand
missed for a few frames keeps its last position; without a hand the saber
hangs straight down from the tip. Drawing only -- the cutting still uses the
pen point alone.
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


def saber_to_hand(tip, hand, hilt_length=None, min_blade=None):
    """Saber from ``tip`` to the hilt centred on ``hand`` (blade at least ``min_blade``)."""
    if hilt_length is None:
        hilt_length = config.SABER_HILT_LENGTH * config.SCALE
    if min_blade is None:
        min_blade = 40 * config.SCALE
    dx, dy = hand[0] - tip[0], hand[1] - tip[1]
    dist = math.hypot(dx, dy)
    d = (dx / dist, dy / dist) if dist > 0 else (0.0, 1.0)
    needed = min_blade + hilt_length / 2
    if dist < needed:                        # hand right at the tip: keep a visible blade
        hand = (tip[0] + d[0] * needed, tip[1] + d[1] * needed)
    half = hilt_length / 2
    as_int = lambda q: (int(round(q[0])), int(round(q[1])))
    return Saber((int(tip[0]), int(tip[1])),
                 as_int((hand[0] - d[0] * half, hand[1] - d[1] * half)),
                 as_int((hand[0] + d[0] * half, hand[1] + d[1] * half)))


class SaberTracker:
    """Saber geometry frame by frame: tip at the pen point, hilt on the hand."""

    def __init__(self, smoothing=0.5, hilt_length=None, keep_frames=5, radius=None):
        self.smoothing = smoothing
        self.hilt_length = hilt_length
        self.keep_frames = keep_frames
        self.radius = radius
        self._offset = None          # hand position relative to the tip (smoothed)
        self._missing = 0

    def update(self, tip, skin, face=None):
        if tip is None:
            self._offset, self._missing = None, 0
            return None
        radius = self.radius if self.radius is not None else config.SABER_HAND_RADIUS * config.SCALE
        hand = find_hand(skin, tip, radius, face)
        if hand is not None:
            offset = (hand[0] - tip[0], hand[1] - tip[1])
            if self._offset is not None:     # smooth the offset: moving pen and hand together
                k = self.smoothing           # does not lag, only jitter is damped
                offset = (k * self._offset[0] + (1 - k) * offset[0],
                          k * self._offset[1] + (1 - k) * offset[1])
            self._offset, self._missing = offset, 0
        else:
            self._missing += 1
            if self._missing > self.keep_frames:
                self._offset = None
        if self._offset is None:
            return saber_at(tip, hilt_length=self.hilt_length)
        hand = (tip[0] + self._offset[0], tip[1] + self._offset[1])
        return saber_to_hand(tip, hand, self.hilt_length)


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
