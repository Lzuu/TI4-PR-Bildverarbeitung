"""Tests for the difficulty presets (lives fixed at 3, speed is the difficulty)."""

import random

from difficulty import LEVELS, EASY, MEDIUM, HARD, DEFAULT
from game import Game


def test_three_levels_in_order():
    assert [lvl.name for lvl in LEVELS] == ["Einfach", "Mittel", "Schwer"]


def test_every_level_starts_a_game_with_the_same_lives():
    # difficulty changes the pace only, never how many misses are allowed
    lives = {Game(960, 540, difficulty=lvl, rng=random.Random(0)).lives for lvl in LEVELS}
    assert len(lives) == 1


def test_higher_difficulty_is_faster():
    # Faster = higher gravity, faster launch, faster spawning
    assert EASY.gravity < MEDIUM.gravity < HARD.gravity
    assert (abs(EASY.launch_vy[0]) < abs(MEDIUM.launch_vy[0])
            < abs(HARD.launch_vy[0]))
    assert (min(EASY.spawn_interval) >= min(MEDIUM.spawn_interval)
            >= min(HARD.spawn_interval))


def test_default_is_medium():
    assert DEFAULT is MEDIUM
