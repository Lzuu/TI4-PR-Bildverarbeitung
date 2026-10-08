"""Tests for fruit projectile physics (thrown up from below, arcs, falls back)."""

import random

import config
from fruit import Fruit

W, H = config.WIDTH, config.HEIGHT


def _make_fruit(seed=0, bomb=False):
    random.seed(seed)
    f = Fruit(W, H)
    f.is_bomb = bomb
    return f


def test_launches_upward_from_below_the_bottom():
    f = _make_fruit()
    assert f.vy < 0, "fruit must start with an upward velocity"
    assert f.y > H, "fruit must start below the bottom edge"


def test_arc_enters_and_peaks_inside_the_frame():
    """Over many launches the peak must be visible (not cut off at the top)."""
    for seed in range(200):
        f = _make_fruit(seed)
        peak = f.y
        entered = False
        for _ in range(400):
            f.update()
            peak = min(peak, f.y)
            if f.y - f.radius < H:
                entered = True
            if f.vy > 0 and f.y - f.radius > H:
                break
        assert entered, "fruit never became visible"
        assert peak > 0, f"fruit flew off the top (peak y={peak:.0f})"


def test_rising_fruit_is_not_missed():
    f = _make_fruit()
    # Right after launch it is still below the edge but rising -> not a miss
    assert not f.is_missed(H)


def test_missed_only_while_descending():
    f = _make_fruit()
    missed = False
    for _ in range(400):
        f.update()
        if f.is_missed(H):
            missed = True
            assert f.vy > 0, "a miss may only trigger while falling back down"
            break
    assert missed, "fruit should eventually fall back out and count as missed"


def test_slice_marks_state_and_stops_scoring_twice():
    f = _make_fruit()
    f.slice(0.0)
    assert f.sliced
    # The slice animation finishes after the configured number of frames
    for _ in range(config.SLICE_ANIM_FRAMES):
        f.update()
    assert f.is_finished()
