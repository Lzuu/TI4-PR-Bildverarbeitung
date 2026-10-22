"""Draw a lightsaber at the tracked pen point.

Deliberately static: the tip of the blade is the point that cuts the fruit, the
blade hangs straight down from it, the hilt sits below. No orientation or hand
tracking -- a following, tilting saber looked jittery and wrong in practice.
Drawing only; the cutting still uses the pen point alone.
"""

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

    # hilt: dark metal grip with ridges, bright emitter ring on top, pommel cap
    width = max(6, int(18 * s))
    x, y_top, y_bottom = emitter[0], emitter[1], pommel[1]
    cv2.rectangle(frame, (x - width // 2, y_top), (x + width // 2, y_bottom), (60, 60, 65), -1)
    for t in (0.35, 0.5, 0.65, 0.8):
        y = int(y_top + (y_bottom - y_top) * t)
        cv2.line(frame, (x - width // 2, y), (x + width // 2, y), (25, 25, 30), max(1, int(3 * s)))
    ring = max(1, int(3 * s))
    cv2.rectangle(frame, (x - width // 2 - ring, y_top),
                  (x + width // 2 + ring, y_top + max(2, int(6 * s))), (190, 190, 195), -1)
    cv2.rectangle(frame, (x - width // 2, y_bottom - max(2, int(5 * s))), (x + width // 2, y_bottom),
                  (120, 120, 125), -1)
    return frame
