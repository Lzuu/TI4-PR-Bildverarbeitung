"""The blade: a short trail of the most recent hand positions.

Two things are kept apart: the *visible* trail (``points``, drawn as a glowing
streak, allowed to fade out after the pen is lost) and the *live* measurements
used for slicing (``_live``). When the pen is lost, the live track ends at once
-- no cut from a stale position -- and a reacquired pen starts a new track, so
no segment is ever drawn across the gap (#41, CR-01).
"""

import math
from collections import deque

import cv2

import config


class Blade:
    def __init__(self):
        self.points = deque(maxlen=config.TRAIL_LENGTH)   # visible trail
        self._live = deque(maxlen=2)                      # current measurements (slicing)

    def reset(self):
        self.points.clear()
        self._live.clear()

    def add(self, point):
        """Add a hand point, or ``None`` when the hand is lost in this frame."""
        if point is None:
            self._live.clear()               # no current measurement: no collision
            if self.points:
                self.points.popleft()        # visible trail fades out
            return
        if not self._live:
            self.points.clear()              # new track: no streak across the gap
        self._live.append(point)
        self.points.append(point)

    def segment(self):
        """Return (prev, cur) of the last two live points, or ``None``."""
        if len(self._live) < 2:
            return None
        return self._live[-2], self._live[-1]

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
        """The current marker point, or ``None`` if the pen is not seen right now."""
        return self._live[-1] if self._live else None

    def draw(self, frame, glow_color=None):
        """Draw the trail; ``glow_color`` (BGR) e.g. matches the lightsaber colour."""
        pts = list(self.points)
        if len(pts) < 2:
            return
        s = config.SCALE
        # Glow pass on a copy, then blend for a soft halo
        overlay = frame.copy()
        for i in range(1, len(pts)):
            thick = int((2 + (i / len(pts)) * config.BLADE_MAX_THICKNESS) * s)
            cv2.line(overlay, pts[i - 1], pts[i], glow_color or config.BLADE_GLOW_COLOR,
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
