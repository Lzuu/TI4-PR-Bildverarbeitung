"""Tests for the fixed pen-colour definitions."""

from colors import COLORS


def test_ranges_are_well_formed():
    for c in COLORS:
        assert len(c.lower) == 3 and len(c.upper) == 3
        for lo, hi in zip(c.lower, c.upper):
            assert 0 <= lo <= hi <= 255
