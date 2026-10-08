"""Small reusable helpers: geometry for collision and text drawing."""

import math

import cv2
import numpy as np

import config


def draw_heart(frame, center, r, color=(70, 70, 235)):
    """Draw a filled heart icon (two lobes + a downward triangle).

    ``r`` is the lobe radius; the heart is roughly 4*r wide and tall.
    """
    cx, cy = int(center[0]), int(center[1])
    r = max(2, int(r))
    cv2.circle(frame, (cx - r, cy - r), r, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx + r, cy - r), r, color, -1, cv2.LINE_AA)
    pts = np.array([[cx - 2 * r, cy - r], [cx + 2 * r, cy - r], [cx, cy + 2 * r]],
                   np.int32)
    cv2.fillPoly(frame, [pts], color, cv2.LINE_AA)


def point_segment_distance(p, a, b):
    """Shortest distance from point ``p`` to the line segment ``a``-``b``.

    Used for blade/fruit collision: the blade is the segment between the last
    two hand positions, the fruit is a circle. If this distance <= radius the
    blade cut through the fruit.
    """
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        # a and b are the same point -> plain point-to-point distance
        return math.hypot(px - ax, py - ay)
    # Project p onto the segment, clamped to [0, 1]
    t = ((px - ax) * dx + (py - ay) * dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    proj_x = ax + t * dx
    proj_y = ay + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def draw_text(frame, text, org, scale=0.8, color=None, thickness=2, center=False):
    """Draw text with a dark shadow for readability over the camera feed."""
    if color is None:
        color = config.HUD_COLOR
    font = cv2.FONT_HERSHEY_SIMPLEX
    if center:
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        org = (int(org[0] - tw / 2), int(org[1] + th / 2))
    x, y = int(org[0]), int(org[1])
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, config.HUD_SHADOW,
                thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
