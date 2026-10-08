"""Tests for the start-menu layout and the mouse colour-swatch hit test."""

import config
import main as m
from colors import COLORS


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
