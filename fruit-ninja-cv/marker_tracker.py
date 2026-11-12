"""Marker tracking via fixed HSV colour ranges (no self-calibration).

The tracker can hold one or several colour ranges at once: a single colour
during play, or the union of all pen colours on the colour-selection screen
(so a pen of any supported colour is detected). Pipeline per frame:

  BGR -> HSV -> inRange(s) (OR-combined) -> morphology -> largest contour
      -> centroid (cv2.moments)

With ``scale`` < 1 the whole pipeline runs on a downscaled copy (#42, PERF-04):
the morphology and blur kernels shrink with it (a fixed 5x5 kernel would erase pens
thinner than ~8 px), the area threshold follows the scale squared, and the
returned point and ``last_mask`` are mapped back to full-frame coordinates.

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

    def __init__(self, ranges=None, point_mode="centroid", scale=None):
        self.point_mode = point_mode     # "centroid" or "tip"
        self.scale = config.TRACK_SCALE if scale is None else scale
        self._dir = None                 # last pen axis direction (tip mode)
        k = config.MORPH_KERNEL
        self._blur = (5, 5)
        if self.scale < 1.0:
            k = max(3, int(round(k * self.scale)) | 1)       # odd, shrinks with the frame
            self._blur = (3, 3)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.last_mask = None            # every pen-coloured pixel of the frame (debug view)
        self.pen_mask = None             # only the contour chosen as the pen (what the suit spares)
        self._no_pen = None              # cached all-zero mask for frames without a pen
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
        """Return the marker point (x, y) in ``frame`` coordinates, or ``None``."""
        full_h, full_w = frame.shape[:2]
        s = self.scale
        if s != 1.0:
            frame = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = None
        for lo, hi in self.ranges:
            m = cv2.inRange(hsv, lo, hi)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)
        mask = cv2.GaussianBlur(mask, self._blur, 0)
        small_mask = mask
        if s != 1.0:
            mask = cv2.resize(mask, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        self.last_mask = mask                # always frame-sized, shows ALL pen-coloured pixels
        if self._no_pen is None or self._no_pen.shape != (full_h, full_w):
            self._no_pen = np.zeros((full_h, full_w), np.uint8)
        self.pen_mask = self._no_pen         # until a pen is found: nothing to protect (#71)
        mask = small_mask

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._dir = None
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < config.MIN_MARKER_AREA * s * s:
            self._dir = None
            return None
        m = cv2.moments(largest)
        if m["m00"] == 0:
            self._dir = None
            return None
        # the pen is this one contour: a pink cushion elsewhere must not be spared by the suit
        selected = np.zeros_like(mask)
        cv2.drawContours(selected, [largest], -1, 255, cv2.FILLED)
        selected = cv2.bitwise_and(mask, cv2.dilate(selected, self._kernel))   # keeps the soft edge
        self.pen_mask = selected if s == 1.0 else cv2.resize(
            selected, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
        centroid = (m["m10"] / m["m00"], m["m01"] / m["m00"])
        if self.point_mode == "tip":
            centroid = self._tip(largest, centroid)
        return (int(round(centroid[0] / s)), int(round(centroid[1] / s)))

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
