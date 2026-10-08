"""Tests for the blade trail (recent marker points, speed, current point)."""

import config
from blade import Blade


def test_empty_blade_has_no_segment_or_point():
    b = Blade()
    assert b.segment() is None
    assert b.current_point() is None
    assert b.speed() == 0.0


def test_segment_speed_and_current_point():
    b = Blade()
    b.add((0, 0))
    b.add((3, 4))
    assert b.segment() == ((0, 0), (3, 4))
    assert b.current_point() == (3, 4)
    assert b.speed() == 5.0            # 3-4-5 triangle


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
