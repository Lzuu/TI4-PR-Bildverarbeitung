"""Draw the tracked pen as a lightsaber.

The tracked pen point -- the point that cuts the fruit -- is the **tip** of the
blade. The blade runs down along the pen to the hand holding it, where the hilt
sits. Per frame:

  pen mask -> largest blob -> centre, length, axis direction (fitLine)
  skin mask near the lower end of the pen -> the hand
  -> blade from the pen point down to the hilt, hilt on the hand.

The axis points "up" by default and then follows the pen continuously (it does
not flip when the pen turns past horizontal). A pen seen end-on (round blob)
keeps the last direction. Direction and hand position are smoothed. Without a
visible hand the hilt sits at the lower end of the pen. Drawing only -- the
cutting still uses the pen point alone.
"""

import math
from collections import namedtuple

import cv2
import numpy as np

import config

PenAxis = namedtuple("PenAxis", "center direction length width")
Saber = namedtuple("Saber", "tip hilt_top hilt_bottom width")

MIN_ELONGATION = 1.8             # length/width below this: no reliable direction


def unit(a, b):
    """Unit vector from point ``b`` to point ``a``."""
    dx, dy = a[0] - b[0], a[1] - b[1]
    n = math.hypot(dx, dy) or 1.0
    return dx / n, dy / n


def pen_axis(mask, min_area=None, min_elongation=MIN_ELONGATION):
    """Centre, direction (unit vector, pointing up), length and width of the pen.

    ``direction`` is None for a roundish blob (pen seen end-on). Returns None
    without a pen (empty mask or only small noise).
    """
    min_area = config.MIN_MARKER_AREA if min_area is None else min_area
    binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    blob = max(contours, key=cv2.contourArea)
    if cv2.contourArea(blob) < min_area:
        return None
    (cx, cy), (w, h), _ = cv2.minAreaRect(blob)
    length, width = max(w, h), min(w, h)
    if length < min_elongation * max(width, 1.0):
        return PenAxis((cx, cy), None, length, width)
    vx, vy = cv2.fitLine(blob, cv2.DIST_L2, 0, 0.01, 0.01).ravel()[:2]
    if vy > 0 or (vy == 0 and vx < 0):           # make it point upwards
        vx, vy = -vx, -vy
    n = math.hypot(vx, vy)
    return PenAxis((cx, cy), (vx / n, vy / n), length, width)


MIN_HAND_PIXELS = 40             # fewer skin pixels near the pen: no hand found


def find_hand(skin, start, radius):
    """Centre of the skin pixels within ``radius`` of ``start`` (the hand), or None."""
    if skin is None:
        return None
    h, w = skin.shape[:2]
    sx, sy = start
    x0, y0 = max(int(sx - radius), 0), max(int(sy - radius), 0)
    x1, y1 = min(int(sx + radius) + 1, w), min(int(sy + radius) + 1, h)
    if x0 >= x1 or y0 >= y1:
        return None
    ys, xs = np.nonzero(skin[y0:y1, x0:x1])
    xs, ys = xs + x0, ys + y0
    near = (xs - sx) ** 2 + (ys - sy) ** 2 <= radius * radius
    if near.sum() < MIN_HAND_PIXELS:
        return None
    return float(xs[near].mean()), float(ys[near].mean())


class SaberTracker:
    """Turns pen mask + pen point (+ skin mask) into saber geometry, frame by frame."""

    def __init__(self, hilt_length=None, smoothing=0.5):
        self.hilt_length = hilt_length
        self.smoothing = smoothing
        self._dir = None
        self._hand = None

    def _smooth(self, old, new):
        if old is None:
            return new
        k = self.smoothing
        return (k * old[0] + (1 - k) * new[0], k * old[1] + (1 - k) * new[1])

    def update(self, mask, point, skin=None):
        """``Saber(tip, hilt_top, hilt_bottom, width)`` as int points, or None.

        ``point`` is the tracked pen point (blade tip), ``skin`` a skin mask to
        find the hand that holds the pen.
        """
        axis = pen_axis(mask) if point is not None else None
        if axis is None:
            self._dir = self._hand = None    # pen gone: start fresh next time
            return None
        d = axis.direction
        cx, cy = axis.center
        towards_point = (point[0] - cx, point[1] - cy)
        if d is not None and math.hypot(*towards_point) > 0.2 * axis.length:
            # the point sits at one end of the pen: the axis points to that end
            if d[0] * towards_point[0] + d[1] * towards_point[1] < 0:
                d = (-d[0], -d[1])
            if self._dir is not None:
                d = self._smooth(self._dir, d)
                n = math.hypot(*d) or 1.0
                d = (d[0] / n, d[1] / n)
        elif d is None:                      # seen end-on: keep the last direction
            d = self._dir or (0.0, -1.0)
        elif self._dir is not None:
            if d[0] * self._dir[0] + d[1] * self._dir[1] < 0:
                d = (-d[0], -d[1])           # continuity beats "up": no flip past horizontal
            d = self._smooth(self._dir, d)
            n = math.hypot(*d) or 1.0
            d = (d[0] / n, d[1] / n)
        self._dir = d

        # the hand holds the lower end of the pen; skin there marks it
        half = axis.length / 2
        lower_end = (cx - d[0] * half, cy - d[1] * half)
        hand = find_hand(skin, lower_end, max(axis.length * 0.75, 30.0)) or lower_end
        hand = self._hand = self._smooth(self._hand, hand)

        up = d if math.dist(point, hand) < 1 else unit(point, hand)
        hilt = self.hilt_length
        if hilt is None:
            hilt = config.SABER_HILT_LENGTH * config.SCALE
        top_offset = min(hilt / 2, math.dist(point, hand) / 2)   # hilt never past the tip
        hilt_top = (hand[0] + up[0] * top_offset, hand[1] + up[1] * top_offset)
        hilt_bottom = (hand[0] - up[0] * hilt / 2, hand[1] - up[1] * hilt / 2)
        as_int = lambda p: (int(round(p[0])), int(round(p[1])))
        return Saber(as_int(point), as_int(hilt_top), as_int(hilt_bottom), axis.width)


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
    """Draw blade (glow + white core) and hilt into ``frame`` (in place)."""
    if saber is None:
        return frame
    s = config.SCALE
    tip, emitter, grip, pen_width = saber            # blade: emitter (hilt top) -> tip
    # blade: wide soft glow, tighter bright glow, white-hot core
    _add_glow(frame, emitter, tip, color, max(2, int(22 * s)), max(1.0, 9 * s))
    _add_glow(frame, emitter, tip, color, max(2, int(10 * s)), max(1.0, 3 * s))
    cv2.line(frame, emitter, tip, (255, 255, 255), max(2, int(5 * s)), cv2.LINE_AA)
    # hilt over the real pen: dark metal with grip ridges and a bright emitter ring
    # hilt on the hand: about pen width, never absurdly wide (pen close to the camera)
    hilt = min(max(3, int(14 * s), int(pen_width) + 6), max(3, int(32 * s)))
    cv2.line(frame, grip, emitter, (60, 60, 65), hilt, cv2.LINE_AA)
    d = unit(emitter, grip)
    perp = (-d[1], d[0])
    r = hilt // 2 + 1
    for t in (0.15, 0.3, 0.45):
        px = grip[0] + (emitter[0] - grip[0]) * t
        py = grip[1] + (emitter[1] - grip[1]) * t
        cv2.line(frame, (int(px - perp[0] * r), int(py - perp[1] * r)),
                 (int(px + perp[0] * r), int(py + perp[1] * r)), (25, 25, 30),
                 max(1, int(3 * s)), cv2.LINE_AA)
    cv2.line(frame, (int(emitter[0] - perp[0] * r), int(emitter[1] - perp[1] * r)),
             (int(emitter[0] + perp[0] * r), int(emitter[1] + perp[1] * r)), (190, 190, 195),
             max(2, int(4 * s)), cv2.LINE_AA)
    return frame
