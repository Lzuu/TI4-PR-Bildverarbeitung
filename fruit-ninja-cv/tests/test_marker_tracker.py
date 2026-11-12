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


# -- pen tip mode: the upper end of the pen is the point (lightsaber tip) -----
def _pen_frame(center=(480, 270), length=180, width=25, angle=0):
    frame = np.full((H, W, 3), WALL, np.uint8)
    box = cv2.boxPoints((center, (width, length), angle)).astype(np.int32)
    cv2.fillPoly(frame, [box], PINK_BGR)
    return frame


def test_tip_mode_returns_the_upper_end_of_the_pen():
    pt = MarkerTracker(point_mode="tip").track(_frame_with(PINK_BGR))
    assert abs(pt[0] - 482) < 8 and abs(pt[1] - 180) < 8


def test_tip_mode_follows_a_tilted_pen():
    import math
    pt = MarkerTracker(point_mode="tip").track(_pen_frame(angle=30))
    top = (480 + 90 * math.sin(math.radians(30)), 270 - 90 * math.cos(math.radians(30)))
    assert math.dist(pt, top) < 10


def test_tip_mode_does_not_jump_to_the_other_end_past_horizontal():
    t = MarkerTracker(point_mode="tip")
    t.track(_pen_frame(angle=80))                    # tip on the right, slightly up
    pt = t.track(_pen_frame(angle=100))              # turned just past horizontal
    assert pt[0] > 480 + 60                          # still the right end


def test_tip_mode_uses_the_centre_of_a_round_marker():
    frame = np.full((H, W, 3), WALL, np.uint8)
    cv2.circle(frame, (480, 270), 25, PINK_BGR, -1)
    pt = MarkerTracker(point_mode="tip").track(frame)
    assert abs(pt[0] - 480) < 4 and abs(pt[1] - 270) < 4


# -- PERF-04 (#42): tracking at a reduced processing scale -------------------
import pytest


def _thin_pen_frame(width, x=300, y=300):
    frame = np.full((H, W, 3), 130, np.uint8)
    cv2.line(frame, (x, y + 70), (x + 12, y), PINK_BGR, width)     # tip at (x + 12, y)
    return frame, (x + 12, y)


@pytest.mark.parametrize("width", [3, 4, 5, 6, 8, 12])
def test_reduced_scale_still_finds_thin_pens_at_the_same_place(width):
    frame, tip = _thin_pen_frame(width)
    full = MarkerTracker(point_mode="tip", scale=1.0).track(frame)
    half = MarkerTracker(point_mode="tip", scale=0.5).track(frame)
    assert full is not None and half is not None
    assert abs(half[0] - full[0]) <= 8 and abs(half[1] - full[1]) <= 8   # full-frame coordinates
    assert abs(half[0] - tip[0]) <= 10 and abs(half[1] - tip[1]) <= 10


def test_reduced_scale_returns_full_size_mask():
    frame, _ = _thin_pen_frame(8)
    tracker = MarkerTracker(point_mode="tip", scale=0.5)
    tracker.track(frame)
    assert tracker.last_mask.shape == frame.shape[:2]
    assert tracker.last_mask[300, 312] > 0 or tracker.last_mask[310, 311] > 0   # pen pixels set


@pytest.mark.parametrize("side,found", [(8, False), (12, False), (18, True), (30, True)])
def test_area_threshold_is_the_same_at_every_scale(side, found):
    # MIN_MARKER_AREA is defined at full resolution: a square blob is accepted from
    # about 15 px side length (the blur grows it) -- at scale 1.0 and 0.5 alike
    frame = np.full((H, W, 3), 130, np.uint8)
    cv2.rectangle(frame, (400, 300), (400 + side - 1, 300 + side - 1), PINK_BGR, -1)
    for scale in (1.0, 0.5):
        assert (MarkerTracker(scale=scale).track(frame) is not None) == found, scale


def test_scale_defaults_to_config(monkeypatch):
    monkeypatch.setattr(config, "TRACK_SCALE", 0.5)
    assert MarkerTracker().scale == 0.5
    monkeypatch.setattr(config, "TRACK_SCALE", 1.0)
    assert MarkerTracker().scale == 1.0
