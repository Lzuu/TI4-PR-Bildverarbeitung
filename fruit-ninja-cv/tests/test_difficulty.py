"""Tests for the difficulty presets (lives fixed at 3, speed is the difficulty)."""

from difficulty import LEVELS, EASY, MEDIUM, HARD, DEFAULT


def test_three_levels_in_order():
    assert [lvl.name for lvl in LEVELS] == ["Einfach", "Mittel", "Schwer"]


def test_lives_always_three():
    assert all(lvl.lives == 3 for lvl in LEVELS)


def test_higher_difficulty_is_faster():
    # Faster = higher gravity, faster launch, faster spawning
    assert EASY.gravity < MEDIUM.gravity < HARD.gravity
    assert (abs(EASY.launch_vy[0]) < abs(MEDIUM.launch_vy[0])
            < abs(HARD.launch_vy[0]))
    assert (min(EASY.spawn_interval) >= min(MEDIUM.spawn_interval)
            >= min(HARD.spawn_interval))


def test_default_is_medium():
    assert DEFAULT is MEDIUM
