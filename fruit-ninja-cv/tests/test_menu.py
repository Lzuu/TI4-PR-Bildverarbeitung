"""Tests for the start-menu layout and the mouse colour-swatch hit test."""

import background
import config
import main as m
from colors import COLORS


def _overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def test_one_swatch_per_colour():
    assert len(m.color_swatches(config.WIDTH, config.HEIGHT)) == len(COLORS)


def test_click_inside_swatch_is_a_hit():
    swatches = m.color_swatches(config.WIDTH, config.HEIGHT)
    for cx, cy, r in swatches:
        assert m._hit((cx, cy), (cx, cy, r))            # centre hits
        assert not m._hit((cx + r + 50, cy), (cx, cy, r))  # far away misses


def test_start_target_is_above_quit_target():
    start = m.menu_start_target(config.WIDTH, config.HEIGHT)
    quit_t = m.quit_target(config.WIDTH, config.HEIGHT)
    assert start[1] < quit_t[1]        # Start sits above Quit, no overlap


def test_background_button_does_not_overlap_other_buttons():
    w, h = config.WIDTH, config.HEIGHT
    assert not _overlap(m.background_button(w, h), m.settings_button(w, h))
    assert not _overlap(m.background_button(w, h), m.camera_button(w, h))


def test_click_on_background_button_cycles_mode():
    w, h = config.WIDTH, config.HEIGHT
    x, y, bw, bh = m.background_button(w, h)
    start = background.MODES[0]
    assert m.handle_background_click((x + 5, y + 5), start, w, h) == background.next_mode(start)


def test_click_outside_background_button_keeps_mode():
    w, h = config.WIDTH, config.HEIGHT
    assert m.handle_background_click((-50, -50), background.MODE_NORMAL, w, h) == background.MODE_NORMAL


# -- CR-04 (#41): green pen and greenscreen are an incompatible choice -------
import background
from colors import GELB, GRUEN, PINK


def test_green_pen_is_blocked_only_in_greenscreen_mode():
    assert not m.pen_allowed(GRUEN, background.MODE_GREENSCREEN)
    for mode in (background.MODE_NORMAL, background.MODE_YOLO, background.MODE_OFF):
        assert m.pen_allowed(GRUEN, mode)


def test_pink_and_yellow_work_with_greenscreen():
    assert m.pen_allowed(PINK, background.MODE_GREENSCREEN)
    assert m.pen_allowed(GELB, background.MODE_GREENSCREEN)


def test_switching_to_greenscreen_replaces_a_green_pen():
    assert m.pen_for_mode(GRUEN, background.MODE_GREENSCREEN) is PINK
    assert m.pen_for_mode(GELB, background.MODE_GREENSCREEN) is GELB
    assert m.pen_for_mode(GRUEN, background.MODE_NORMAL) is GRUEN
