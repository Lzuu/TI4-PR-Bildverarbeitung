"""The blade: a short trail of the most recent hand positions.

The segment between the two most recent points is used for slicing; the whole
trail is drawn as a glowing streak.
"""

import math
from collections import deque

import cv2

import config


class Blade:
    def __init__(self):
        self.points = deque(maxlen=config.TRAIL_LENGTH)

    def reset(self):
        self.points.clear()

    def add(self, point):
        """Add a hand point, or ``None`` to break the trail when the hand is lost."""
        if point is None:
            # Lose the trail quickly so a reappearing hand does not "teleport-cut"
            if self.points:
                self.points.popleft()
            return
        self.points.append(point)

    def segment(self):
        """Return (prev, cur) of the last two points, or ``None``."""
        if len(self.points) < 2:
            return None
        return self.points[-2], self.points[-1]

    def speed(self):
        """Pixel distance covered in the last step (0 if not enough points)."""
        seg = self.segment()
        if seg is None:
            return 0.0
        (ax, ay), (bx, by) = seg
        return math.hypot(bx - ax, by - ay)

    def angle(self):
        """Direction of the last movement step in radians."""
        seg = self.segment()
        if seg is None:
            return 0.0
        (ax, ay), (bx, by) = seg
        return math.atan2(by - ay, bx - ax)

    def current_point(self):
        """The most recent marker point, or ``None`` if the trail is empty."""
        return self.points[-1] if self.points else None

    def draw(self, frame):
        pts = list(self.points)
        if len(pts) < 2:
            return
        s = config.SCALE
        # Glow pass on a copy, then blend for a soft halo
        overlay = frame.copy()
        for i in range(1, len(pts)):
            thick = int((2 + (i / len(pts)) * config.BLADE_MAX_THICKNESS) * s)
            cv2.line(overlay, pts[i - 1], pts[i], config.BLADE_GLOW_COLOR,
                     thick + int(8 * s), cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
        # Sharp bright core
        for i in range(1, len(pts)):
            thick = max(1, int((2 + (i / len(pts)) * config.BLADE_MAX_THICKNESS) * s))
            cv2.line(frame, pts[i - 1], pts[i], config.BLADE_COLOR,
                     thick, cv2.LINE_AA)
        # Tip marker
        cv2.circle(frame, pts[-1], max(3, int(6 * s)), config.BLADE_COLOR, -1,
                   cv2.LINE_AA)
