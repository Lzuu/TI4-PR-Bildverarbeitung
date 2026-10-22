"""Tests for the static lightsaber drawn at the cutting point. Headless."""

import numpy as np

import lightsaber

POINT = (300, 150)                                      # tracked pen point (cuts the fruit)


def test_no_saber_without_pen_point():
    assert lightsaber.saber_at(None) is None


def test_tip_is_the_cutting_point_and_the_saber_hangs_straight_down():
    s = lightsaber.saber_at(POINT, blade_length=120, hilt_length=50)
    assert s.tip == POINT
    assert s.hilt_top == (300, 150 + 120)                # blade straight down
    assert s.hilt_bottom == (300, 150 + 120 + 50)        # hilt below the blade


def test_default_lengths_scale_with_the_screen(monkeypatch):
    import config
    monkeypatch.setattr(config, "SCALE", 2.0)
    s = lightsaber.saber_at(POINT)
    assert s.hilt_top[1] - s.tip[1] == round(config.SABER_LENGTH * 2.0)


def _drawn(frame=None, color=(255, 0, 0)):
    frame = np.zeros((500, 600, 3), np.uint8) if frame is None else frame
    s = lightsaber.saber_at(POINT, blade_length=150, hilt_length=60)
    lightsaber.draw_saber(frame, s, color)
    return frame, s


def test_blade_is_a_thick_bright_beam():
    frame, s = _drawn()
    y = (s.tip[1] + s.hilt_top[1]) // 2
    assert frame[y, 300 - 5:300 + 6].min() > 200          # white-hot core at least 11 px wide


def test_blade_glows_in_the_saber_colour_around_the_core():
    frame, s = _drawn()
    y = (s.tip[1] + s.hilt_top[1]) // 2
    glow = frame[y, 300 + 16]                              # beside the core
    assert glow[0] > 80 and glow[0] > glow[2] + 40         # blue glow (BGR)
    assert (frame[480, 20] == 0).all()                     # far away untouched


def test_hilt_is_dark_metal_below_the_blade():
    frame, s = _drawn(np.full((500, 600, 3), 200, np.uint8))
    y = (s.hilt_top[1] + s.hilt_bottom[1]) // 2
    assert frame[y, 300].mean() < 120


def test_drawing_near_the_edge_or_without_pen_does_not_crash():
    frame = np.zeros((500, 600, 3), np.uint8)
    lightsaber.draw_saber(frame, lightsaber.saber_at((10, 480)), (0, 0, 255))
    lightsaber.draw_saber(frame, None, (0, 0, 255))
