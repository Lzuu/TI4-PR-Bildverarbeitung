"""Tests for the pink-marker HSV tracking and calibration."""

import cv2
import numpy as np

import config
from marker_tracker import MarkerTracker

W, H = config.WIDTH, config.HEIGHT

# BGR colours for synthetic frames
PINK = (200, 0, 255)      # vivid pink/magenta  (hue ~165)
ORANGE = (0, 120, 255)    # vivid orange        (hue ~14)
WALL = 235                # light grey/white background


def _frame_with(colour, box=((470, 180), (495, 360))):
    frame = np.full((H, W, 3), WALL, np.uint8)
    cv2.rectangle(frame, box[0], box[1], colour, -1)
    return frame


def test_default_range_tracks_pink():
    frame = _frame_with(PINK)
    pt = MarkerTracker().track(frame)
    assert pt is not None
    assert abs(pt[0] - 482) < 20 and abs(pt[1] - 270) < 20


def test_default_pink_range_rejects_orange():
    frame = _frame_with(ORANGE)
    assert MarkerTracker().track(frame) is None


def test_calibrate_from_point_sets_pink_hue():
    frame = _frame_with(PINK)
    mt = MarkerTracker()
    hue = mt.calibrate_from_point(frame, 482, 270)
    assert 150 <= hue <= 175
    assert mt.track(frame) is not None


def test_box_calibration_ignores_background_wall():
    # Thin pink stripe through the calibration box, rest is white wall
    x, y, bw, bh = MarkerTracker.calib_box_px(W, H)
    frame = np.full((H, W, 3), WALL, np.uint8)
    cv2.rectangle(frame, (x + bw // 2 - 8, y), (x + bw // 2 + 8, y + bh), PINK, -1)
    mt = MarkerTracker()
    hue = mt.calibrate(frame)
    assert 150 <= hue <= 175, f"calibrated hue drifted toward the wall: {hue}"
    assert mt.track(frame) is not None


def test_blank_wall_detects_nothing():
    frame = np.full((H, W, 3), WALL, np.uint8)
    assert MarkerTracker().track(frame) is None
