"""Marker tracking via fixed HSV colour ranges (no self-calibration).

The tracker can hold one or several colour ranges at once: a single colour
during play, or the union of all pen colours on the colour-selection screen
(so a pen of any supported colour is detected). Pipeline per frame:

  BGR -> HSV -> inRange(s) (OR-combined) -> morphology -> largest contour
      -> centroid (cv2.moments)

With ``point_mode="tip"`` the returned point is the pen's upper end instead of
the centroid (the end farthest along the pen's axis). The axis keeps its last
orientation, so the point does not jump to the other end when the pen turns
past horizontal. A roundish blob (pen seen end-on, or a dot marker) still
returns the centroid.
"""

import math

import cv2
import numpy as np

import config
from colors import DEFAULT


class MarkerTracker:
    TIP_MIN_ELONGATION = 1.8         # below this length/width ratio: use the centroid

    def __init__(self, ranges=None, point_mode="centroid"):
        self.point_mode = point_mode     # "centroid" or "tip"
        self._dir = None                 # last pen axis direction (tip mode)
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
            self._dir = None
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < config.MIN_MARKER_AREA:
            self._dir = None
            return None
        m = cv2.moments(largest)
        if m["m00"] == 0:
            self._dir = None
            return None
        centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        if self.point_mode == "tip":
            centroid = self._tip(largest, centroid)
        return (int(centroid[0]), int(centroid[1]))

    def _tip(self, contour, centroid):
        """Upper end of an elongated pen (continuous orientation), else the centroid."""
        (_, (w, h), _) = cv2.minAreaRect(contour)
        if max(w, h) < self.TIP_MIN_ELONGATION * max(min(w, h), 1.0):
            return centroid
        vx, vy = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01).ravel()[:2]
        if self._dir is not None:
            if vx * self._dir[0] + vy * self._dir[1] < 0:
                vx, vy = -vx, -vy        # keep the same end as before
        elif vy > 0 or (vy == 0 and vx < 0):
            vx, vy = -vx, -vy            # first sight: the upper end
        n = math.hypot(vx, vy)
        self._dir = (vx / n, vy / n)
        pts = contour.reshape(-1, 2).astype(np.float32)
        proj = (pts[:, 0] - centroid[0]) * self._dir[0] + (pts[:, 1] - centroid[1]) * self._dir[1]
        # middle of the end, not a corner: average all points within a few px of it
        end = pts[proj >= proj.max() - max(3.0, 0.15 * min(w, h))]
        return (float(end[:, 0].mean()), float(end[:, 1].mean()))
