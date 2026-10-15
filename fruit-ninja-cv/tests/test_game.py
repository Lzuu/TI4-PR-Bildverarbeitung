"""Tests for the game logic: touch slicing, bombs, misses, spawning, reset."""

import random

import config
from game import Game
from fruit import Fruit
from blade import Blade

W, H = config.WIDTH, config.HEIGHT


def _fruit_at(x, y, r=45, bomb=False):
    f = Fruit(W, H)
    f.is_bomb = bomb
    f.x, f.y, f.radius = float(x), float(y), r
    f.vy = 0.0
    return f


def test_touch_slices_without_movement():
    """A resting marker on a fruit is enough -- no minimum speed required."""
    g = Game(W, H)
    g.fruits = [_fruit_at(W / 2, H / 2)]
    b = Blade()
    b.add((int(W / 2), int(H / 2)))        # single point, zero speed
    g._handle_slicing(b)
    assert g.fruits[0].sliced
    assert g.score == 1


def test_marker_far_from_fruit_does_not_slice():
    g = Game(W, H)
    g.fruits = [_fruit_at(100, 100, r=40)]
    b = Blade()
    b.add((600, 400))
    g._handle_slicing(b)
    assert not g.fruits[0].sliced
    assert g.score == 0


def test_slicing_bomb_ends_game():
    g = Game(W, H)
    g.fruits = [_fruit_at(W / 2, H / 2, bomb=True)]
    b = Blade()
    b.add((int(W / 2), int(H / 2)))
    g._handle_slicing(b)
    assert g.game_over


def test_missed_fruit_costs_a_life():
    g = Game(W, H)
    f = _fruit_at(W / 2, H + 100)
    f.vy = 5.0                              # descending, below the bottom
    g.fruits = [f]
    before = g.lives
    g.update(Blade())
    assert g.lives == before - 1
    assert g.fruits == []                   # removed after the miss


def test_missing_a_bomb_does_not_cost_a_life():
    g = Game(W, H)
    f = _fruit_at(W / 2, H + 100, bomb=True)
    f.vy = 5.0
    g.fruits = [f]
    before = g.lives
    g.update(Blade())
    assert g.lives == before                # a dropped bomb is fine


def test_spawn_throws_fruits_over_time():
    random.seed(0)
    g = Game(W, H)
    b = Blade()
    for _ in range(max(g.difficulty.spawn_interval) + 5):
        g.update(b)
    assert len(g.fruits) >= 1


def test_reset_restores_initial_state():
    g = Game(W, H)
    g.score = 10
    g.lives = 0
    g.game_over = True
    g.fruits = [_fruit_at(1, 1)]
    g.reset()
    assert g.score == 0
    assert g.lives == g.difficulty.lives
    assert not g.game_over
    assert g.fruits == []


def test_bomb_stops_scoring_in_same_frame():
    """Hitting a bomb ends the run immediately -- no extra fruit scores after it."""
    g = Game(W, H)
    bomb = _fruit_at(W / 2, H / 2, bomb=True)
    fruit = _fruit_at(W / 2, H / 2)      # same spot, would be hit too
    g.fruits = [bomb, fruit]
    b = Blade()
    b.add((int(W / 2), int(H / 2)))
    g._handle_slicing(b)
    assert g.game_over
    assert g.score == 0                  # the bomb came first, loop broke


def test_combo_bonus_awarded_for_three_in_window():
    g = Game(W, H)
    g.fruits = [_fruit_at(W / 2, H / 2) for _ in range(3)]
    b = Blade()
    b.add((int(W / 2), int(H / 2)))
    sliced = g._handle_slicing(b)
    assert sliced == 3 and g.score == 3
    g._update_combo(sliced)
    for _ in range(config.COMBO_WINDOW_FRAMES):   # let the combo window elapse
        g._update_combo(0)
    assert g.score == 3 + 3 * config.COMBO_BONUS_PER_FRUIT
    assert g.combo_text.startswith("COMBO")
    assert g.combo_banner_frames > 0


def test_no_combo_below_minimum():
    g = Game(W, H)
    g.fruits = [_fruit_at(W / 2, H / 2) for _ in range(2)]
    b = Blade()
    b.add((int(W / 2), int(H / 2)))
    sliced = g._handle_slicing(b)
    assert sliced == 2 and g.score == 2
    g._update_combo(sliced)
    for _ in range(config.COMBO_WINDOW_FRAMES):
        g._update_combo(0)
    assert g.score == 2                  # no bonus below COMBO_MIN
    assert g.combo_banner_frames == 0
