"""Tests for the heart sprite icon."""

import numpy as np

from heart import get_heart_sprite, draw_heart


def test_sprite_is_bgra_of_requested_size():
    s = get_heart_sprite(40)
    assert s.shape == (40, 40, 4)
    # Has both fully opaque and fully transparent pixels (a real shape + bg)
    assert s[:, :, 3].max() == 255
    assert s[:, :, 3].min() == 0


def test_sprite_is_cached():
    assert get_heart_sprite(32) is get_heart_sprite(32)


def test_draw_heart_blends_onto_frame():
    frame = np.zeros((120, 120, 3), np.uint8)
    draw_heart(frame, (60, 60), 48)
    assert frame.sum() > 0            # something was drawn


def test_draw_heart_clips_at_edge_without_crashing():
    frame = np.zeros((20, 20, 3), np.uint8)
    draw_heart(frame, (0, 0), 48)     # mostly off-screen
    assert frame.shape == (20, 20, 3)
