"""Tests for the collision geometry helper (utils.point_segment_distance)."""

import pytest

from utils import point_segment_distance


def test_distance_to_segment_perpendicular():
    # Point one unit above the middle of a horizontal segment
    assert point_segment_distance((0, 0), (-5, 1), (5, 1)) == pytest.approx(1.0)


def test_distance_clamps_beyond_endpoint():
    # Point is past the right end -> distance is to that endpoint (5 units)
    assert point_segment_distance((10, 0), (-5, 0), (5, 0)) == pytest.approx(5.0)


def test_degenerate_segment_is_point_distance():
    # a == b -> plain point-to-point distance (3-4-5 triangle)
    assert point_segment_distance((3, 4), (0, 0), (0, 0)) == pytest.approx(5.0)


def test_point_on_segment_is_zero():
    assert point_segment_distance((2, 0), (-5, 0), (5, 0)) == pytest.approx(0.0)
