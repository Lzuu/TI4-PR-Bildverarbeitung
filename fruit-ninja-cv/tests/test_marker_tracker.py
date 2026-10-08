"""Tests for the fixed-colour marker tracking (pink / yellow / green, union)."""

import cv2
import numpy as np

import config
from marker_tracker import MarkerTracker
from colors import PINK, GELB, GRUEN, COLORS

W, H = config.WIDTH, config.HEIGHT

# BGR colours for synthetic frames
PINK_BGR = (200, 0, 255)      # hue ~156
YELLOW_BGR = (0, 230, 230)    # hue ~30
GREEN_BGR = (0, 200, 0)       # hue ~60
ORANGE_BGR = (0, 120, 255)    # hue ~14 (not a pen colour)
WALL = 235


def _frame_with(colour, box=((470, 180), (495, 360))):
    frame = np.full((H, W, 3), WALL, np.uint8)
    cv2.rectangle(frame, box[0], box[1], colour, -1)
    return frame


def _centroid_near(pt, x=482, y=270, tol=25):
    return pt is not None and abs(pt[0] - x) < tol and abs(pt[1] - y) < tol


def test_default_tracks_pink():
    assert _centroid_near(MarkerTracker().track(_frame_with(PINK_BGR)))


def test_default_pink_rejects_orange():
    assert MarkerTracker().track(_frame_with(ORANGE_BGR)) is None


def test_set_color_yellow_and_green():
    t = MarkerTracker()
    t.set_color(GELB)
    assert _centroid_near(t.track(_frame_with(YELLOW_BGR)))
    t.set_color(GRUEN)
    assert _centroid_near(t.track(_frame_with(GREEN_BGR)))


def test_union_detects_any_pen_colour():
    t = MarkerTracker()
    t.set_colors(COLORS)
    for bgr in (PINK_BGR, YELLOW_BGR, GREEN_BGR):
        assert _centroid_near(t.track(_frame_with(bgr))), bgr


def test_blank_wall_detects_nothing():
    frame = np.full((H, W, 3), WALL, np.uint8)
    assert MarkerTracker().track(frame) is None
