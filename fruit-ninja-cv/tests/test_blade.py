"""Tests for the blade trail (recent marker points, segment, current point)."""

import cv2
import numpy as np
import pytest

import config
from blade import Blade


def test_empty_blade_has_no_segment_or_point():
    b = Blade()
    assert b.segment() is None
    assert b.current_point() is None


def test_segment_and_current_point():
    b = Blade()
    b.add((0, 0))
    b.add((3, 4))
    assert b.segment() == ((0, 0), (3, 4))
    assert b.current_point() == (3, 4)


def test_trail_respects_max_length():
    b = Blade()
    for i in range(config.TRAIL_LENGTH + 10):
        b.add((i, 0))
    assert len(b.points) == config.TRAIL_LENGTH
    assert b.current_point() == (config.TRAIL_LENGTH + 9, 0)


def test_adding_none_drops_oldest_point():
    b = Blade()
    b.add((1, 1))
    b.add((2, 2))
    b.add(None)                        # marker lost -> trail shrinks
    assert len(b.points) == 1


def test_reset_clears_trail():
    b = Blade()
    b.add((1, 1))
    b.reset()
    assert b.current_point() is None


def test_trail_glow_can_take_the_saber_colour():
    import numpy as np
    b = Blade()
    for x in range(50, 250, 20):
        b.add((x, 100))
    frame = np.zeros((200, 300, 3), np.uint8)
    b.draw(frame, glow_color=(0, 0, 255))            # red glow (BGR)
    halo = frame[100 - 8, 150]                       # beside the white core
    assert halo[2] > 40 and halo[2] > halo[0] + 20


# -- PERF-14 (#61): the glow pass works only in the trail's rectangle ----------
def _full_frame_draw(frame, points, glow_color=None):
    """The former whole-frame version of Blade.draw, kept as the test oracle."""
    pts = list(points)
    s = config.SCALE
    overlay = frame.copy()
    for i in range(1, len(pts)):
        thick = int((2 + (i / len(pts)) * config.BLADE_MAX_THICKNESS) * s)
        cv2.line(overlay, pts[i - 1], pts[i], glow_color or config.BLADE_GLOW_COLOR,
                 thick + int(8 * s), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    for i in range(1, len(pts)):
        thick = max(1, int((2 + (i / len(pts)) * config.BLADE_MAX_THICKNESS) * s))
        cv2.line(frame, pts[i - 1], pts[i], config.BLADE_COLOR, thick, cv2.LINE_AA)
    cv2.circle(frame, pts[-1], max(3, int(6 * s)), config.BLADE_COLOR, -1, cv2.LINE_AA)
    return frame


@pytest.mark.parametrize("points", [
    [(100, 100), (140, 120), (190, 160), (230, 230)],     # middle of the picture
    [(2, 3), (40, 5), (80, 30)],                          # along the top-left edge
    [(300, 230), (318, 238), (319, 239)],                 # into the bottom-right corner
])
def test_trail_in_its_rectangle_matches_the_whole_frame_drawing(points):
    frame = np.random.default_rng(3).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    b = Blade()
    for p in points:
        b.add(p)
    out = frame.copy()
    b.draw(out, (40, 200, 255))
    assert np.array_equal(out, _full_frame_draw(frame.copy(), points, (40, 200, 255)))
