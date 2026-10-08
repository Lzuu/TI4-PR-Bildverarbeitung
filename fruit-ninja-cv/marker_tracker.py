"""Marker tracking: follow an orange pen (the "sword") via HSV colour segmentation.

Pipeline per frame (classic OpenCV):
  BGR -> HSV -> cv2.inRange(lower, upper) -> morphology (open+close)
      -> largest contour -> centroid (cv2.moments)

A sensible default orange range (config.MARKER_HSV_LOWER/UPPER) is used out of the
box. Optional calibration samples the mean HSV inside the ROI box (hold the pen tip
there and press SPACE) and rebuilds the range as mean +/- HSV_TOLERANCE, which makes
tracking robust to the specific pen and lighting.
"""

import cv2
import numpy as np

import config


class MarkerTracker:
    def __init__(self):
        # Start from the default orange range so the game works without calibration
        self.lower = np.array(config.MARKER_HSV_LOWER, dtype=np.uint8)
        self.upper = np.array(config.MARKER_HSV_UPPER, dtype=np.uint8)
        k = config.MORPH_KERNEL
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        self.last_mask = None      # kept for the optional debug view

    # -- calibration --------------------------------------------------------
    @staticmethod
    def calib_box_px(width, height):
        """Return the calibration ROI (x, y, w, h) in pixels."""
        fx, fy, fw, fh = config.CALIB_BOX
        return (int(fx * width), int(fy * height),
                int(fw * width), int(fh * height))

    def _set_range_from_hue(self, hue):
        """Build an HSV range around a hue; floors on S/V ignore the pale wall."""
        h = float(hue)
        self.lower = np.array(
            (max(0, h - config.H_TOLERANCE), config.S_FLOOR, config.V_FLOOR),
            dtype=np.uint8)
        self.upper = np.array(
            (min(179, h + config.H_TOLERANCE), 255, 255), dtype=np.uint8)

    def calibrate_from_point(self, frame, x, y):
        """Calibrate from a small patch around a clicked point (most precise)."""
        h, w = frame.shape[:2]
        p = config.CALIB_PATCH
        x0, x1 = max(0, x - p), min(w, x + p + 1)
        y0, y1 = max(0, y - p), min(h, y + p + 1)
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hue = float(np.median(hsv.reshape(-1, 3)[:, 0]))
        self._set_range_from_hue(hue)
        return hue

    def calibrate(self, frame):
        """Calibrate from the ROI box, using only saturated pixels (ignore wall)."""
        h, w = frame.shape[:2]
        x, y, bw, bh = self.calib_box_px(w, h)
        hsv = cv2.cvtColor(frame[y:y + bh, x:x + bw], cv2.COLOR_BGR2HSV)
        pts = hsv.reshape(-1, 3)
        # Keep only colourful pixels so the white/grey background is ignored
        colourful = pts[(pts[:, 1] > 60) & (pts[:, 2] > 60)]
        sample = colourful if len(colourful) > 20 else pts
        hue = float(np.median(sample[:, 0]))
        self._set_range_from_hue(hue)
        return hue

    # -- per-frame tracking -------------------------------------------------
    def track(self, frame):
        """Return the marker centroid (x, y) or ``None`` if it is not found."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        # Open removes speckle noise, close fills small holes in the blob
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
        cx = int(m["m10"] / m["m00"])
        cy = int(m["m01"] / m["m00"])
        return (cx, cy)
