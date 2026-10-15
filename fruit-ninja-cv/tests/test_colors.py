"""Tests for the fixed pen-colour definitions."""

from colors import COLORS, PINK, DEFAULT


def test_three_named_colours():
    assert [c.name for c in COLORS] == ["Pink", "Gelb", "Gruen"]


def test_default_is_pink():
    assert DEFAULT is PINK


def test_ranges_are_well_formed():
    for c in COLORS:
        assert len(c.lower) == 3 and len(c.upper) == 3
        for lo, hi in zip(c.lower, c.upper):
            assert 0 <= lo <= hi <= 255
