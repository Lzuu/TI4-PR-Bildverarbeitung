"""Small reusable helpers: geometry for collision and text drawing."""

import math

import cv2

import config


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
