"""Marker tracking via fixed HSV colour ranges (no self-calibration).

The tracker can hold one or several colour ranges at once: a single colour
during play, or the union of all pen colours on the colour-selection screen
(so a pen of any supported colour is detected). Pipeline per frame:

  BGR -> HSV -> inRange(s) (OR-combined) -> morphology -> largest contour
      -> centroid (cv2.moments)
"""

import cv2
import numpy as np

import config
from colors import DEFAULT


class MarkerTracker:
    def __init__(self, ranges=None):
        k = config.MORPH_KERNEL
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.last_mask = None
        self.set_ranges(ranges or [(DEFAULT.lower, DEFAULT.upper)])

    # -- colour configuration ----------------------------------------------
    def set_ranges(self, ranges):
        """Set the active HSV ranges as a list of (lower, upper) tuples."""
        self.ranges = [(np.array(lo, np.uint8), np.array(hi, np.uint8))
                       for lo, hi in ranges]

    def set_color(self, pen):
        """Track a single PenColor."""
        self.set_ranges([(pen.lower, pen.upper)])

    def set_colors(self, pens):
        """Track the union of several PenColors (used on the colour screen)."""
        self.set_ranges([(p.lower, p.upper) for p in pens])

    # -- per-frame tracking -------------------------------------------------
    def track(self, frame):
        """Return the marker centroid (x, y) or ``None`` if it is not found."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in self.ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        self.last_mask = mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < config.MIN_MARKER_AREA:
            return None
        m = cv2.moments(largest)
        if m["m00"] == 0:
            return None
        return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))
