"""Draw the tracked pen as a lightsaber.

The marker tracker already segments the pen by colour. Its mask also tells the
pen's *orientation*: the largest blob is elongated, and its main axis
(``cv2.fitLine``) is the direction the pen points. Per frame:

  pen mask -> largest blob -> centre, length, axis direction
  -> hilt drawn over the real pen, glowing blade from its upper end onwards.

The blade points "up" by default and then follows the pen continuously (it does
not flip when the pen turns past horizontal). A pen seen end-on (round blob)
keeps the last direction. The direction is smoothed against jitter.
"""

import math
from collections import namedtuple

import cv2
import numpy as np

import config

PenAxis = namedtuple("PenAxis", "center direction length width")
Saber = namedtuple("Saber", "grip emitter tip width")

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


class SaberTracker:
    """Turns the pen mask into saber geometry, frame by frame."""

    def __init__(self, blade_length=None, smoothing=0.5):
        self.blade_length = blade_length
        self.smoothing = smoothing
        self._dir = None

    def update(self, mask):
        """``Saber(grip, emitter, tip, width)`` (int points, pen width), or None."""
        axis = pen_axis(mask)
        if axis is None:
            self._dir = None             # pen gone: start fresh next time
            return None
        d = axis.direction
        if d is None:                    # seen end-on: keep the last direction
            d = self._dir or (0.0, -1.0)
        elif self._dir is not None:
            if d[0] * self._dir[0] + d[1] * self._dir[1] < 0:
                d = (-d[0], -d[1])       # continuity beats "up": no flip past horizontal
            k = self.smoothing
            d = (k * self._dir[0] + (1 - k) * d[0], k * self._dir[1] + (1 - k) * d[1])
            n = math.hypot(*d) or 1.0
            d = (d[0] / n, d[1] / n)
        self._dir = d

        blade = self.blade_length
        if blade is None:
            blade = config.SABER_LENGTH * config.SCALE
        cx, cy = axis.center
        half = axis.length / 2
        grip = (cx - d[0] * half, cy - d[1] * half)
        emitter = (cx + d[0] * half, cy + d[1] * half)
        tip = (emitter[0] + d[0] * blade, emitter[1] + d[1] * blade)
        as_int = lambda p: (int(round(p[0])), int(round(p[1])))
        return Saber(as_int(grip), as_int(emitter), as_int(tip), axis.width)


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
    grip, emitter, tip, pen_width = saber
    # blade: wide soft glow, tighter bright glow, white-hot core
    _add_glow(frame, emitter, tip, color, max(2, int(22 * s)), max(1.0, 9 * s))
    _add_glow(frame, emitter, tip, color, max(2, int(10 * s)), max(1.0, 3 * s))
    cv2.line(frame, emitter, tip, (255, 255, 255), max(2, int(5 * s)), cv2.LINE_AA)
    # hilt over the real pen: dark metal with grip ridges and a bright emitter ring
    # covers the real pen, but never absurdly wide (pen very close to the camera)
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
