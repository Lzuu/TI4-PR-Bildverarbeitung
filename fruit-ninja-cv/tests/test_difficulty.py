"""Tests for the difficulty presets."""

from difficulty import LEVELS, EASY, MEDIUM, HARD, DEFAULT


def test_three_levels_in_order():
    assert len(LEVELS) == 3
    assert [lvl.name for lvl in LEVELS] == ["Einfach", "Mittel", "Schwer"]


def test_harder_levels_are_actually_harder():
    # Fewer lives, more bombs, faster spawning as difficulty rises
    assert EASY.lives > MEDIUM.lives > HARD.lives
    assert EASY.bomb_probability < MEDIUM.bomb_probability < HARD.bomb_probability
    assert (min(EASY.spawn_interval) >= min(MEDIUM.spawn_interval)
            >= min(HARD.spawn_interval))


def test_default_is_medium():
    assert DEFAULT is MEDIUM
