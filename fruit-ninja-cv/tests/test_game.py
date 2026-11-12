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


def test_marker_far_from_fruit_does_not_slice():
    g = Game(W, H)
    g.fruits = [_fruit_at(100, 100, r=40)]
    b = Blade()
    b.add((600, 400))
    g._handle_slicing(b)
    assert not g.fruits[0].sliced
    assert g.score == 0


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
    g._update_combo(0, config.COMBO_WINDOW_SECONDS)   # let the combo window elapse
    assert g.score == 3 + 3 * config.COMBO_BONUS_PER_FRUIT
    assert g.combo_text.startswith("COMBO")
    assert g.combo_banner_left > 0


def test_no_combo_below_minimum():
    g = Game(W, H)
    g.fruits = [_fruit_at(W / 2, H / 2) for _ in range(2)]
    b = Blade()
    b.add((int(W / 2), int(H / 2)))
    sliced = g._handle_slicing(b)
    assert sliced == 2 and g.score == 2
    g._update_combo(sliced)
    g._update_combo(0, config.COMBO_WINDOW_SECONDS)
    assert g.score == 2                  # no bonus below COMBO_MIN
    assert g.combo_banner_left == 0


# -- CR-01 (#41): tracking loss must interrupt collisions immediately --------
def _trail(blade, points):
    for p in points:
        blade.add(p)


def test_no_cut_while_the_pen_is_lost():
    g = Game(W, H)
    b = Blade()
    _trail(b, [(100 + i, 100) for i in range(12)])          # last point (111, 100)
    g.fruits = [_fruit_at(111, 100, r=30)]                   # fruit at the last position
    for _ in range(3):
        b.add(None)                                          # pen lost
        g._handle_slicing(b)
    assert not g.fruits[0].sliced and g.score == 0


def test_reacquired_pen_does_not_cut_across_the_gap():
    g = Game(W, H)
    b = Blade()
    _trail(b, [(100 + i, 100) for i in range(12)])
    b.add(None)
    g.fruits = [_fruit_at(300, 100, r=30)]                   # between old and new position
    b.add((500, 100))                                        # pen found again far away
    g._handle_slicing(b)
    assert not g.fruits[0].sliced and g.score == 0


def test_cutting_works_again_after_reacquiring():
    g = Game(W, H)
    b = Blade()
    _trail(b, [(100, 100), (110, 100)])
    b.add(None)
    b.add((500, 100))
    b.add((560, 100))                                        # a real swipe after the gap
    g.fruits = [_fruit_at(530, 100, r=20)]
    g._handle_slicing(b)
    assert g.fruits[0].sliced


def test_visible_trail_fades_after_loss_but_does_not_collide():
    b = Blade()
    _trail(b, [(100 + 10 * i, 100) for i in range(6)])
    b.add(None)
    assert len(b.points) > 0                                 # still drawn, fading
    assert b.current_point() is None and b.segment() is None # but no collision


# -- PERF-01 (#42): game time in seconds, independent of the frame rate ------
import pytest

from difficulty import MEDIUM


def _simulate(fps, seconds, seed=1, time_scale=1.0):
    g = Game(W, H, MEDIUM, rng=random.Random(seed))
    g.time_scale = time_scale
    b = Blade()
    for _ in range(round(seconds * fps)):
        g.update(b, 1.0 / fps)
    return g


@pytest.mark.parametrize("fps", [15, 60])
def test_same_time_gives_same_throws_at_any_frame_rate(fps):
    ref = _simulate(30, 3.0)
    g = _simulate(fps, 3.0)
    assert ref.fruits, "test needs fruits in the air"
    assert len(g.fruits) == len(ref.fruits)
    for a, b in zip(g.fruits, ref.fruits):
        assert (a.name, a.is_bomb) == (b.name, b.is_bomb)
        assert abs(a.x - b.x) < 1.0 and abs(a.y - b.y) < 1.0


@pytest.mark.parametrize("fps", [10, 30, 60])
def test_combo_window_is_in_seconds(fps):
    g = Game(W, H, rng=random.Random(0))
    g.fruits = [_fruit_at(W / 2, H / 2) for _ in range(3)]
    b = Blade()
    b.add((int(W / 2), int(H / 2)))
    g.update(b, 1.0 / fps)                                   # three cut at once
    assert g.score == 3
    empty = Blade()
    t = 0.0
    while t + 1.0 / fps < config.COMBO_WINDOW_SECONDS - 1e-9:
        g.update(empty, 1.0 / fps)
        t += 1.0 / fps
    assert g.score == 3                                      # window still open
    g.update(empty, 1.0 / fps)
    g.update(empty, 1.0 / fps)
    assert g.score == 3 + 3 * config.COMBO_BONUS_PER_FRUIT


def test_slow_motion_only_slows_the_simulation():
    slow = _simulate(30, 2.0, time_scale=0.5)
    normal = _simulate(30, 1.0)
    assert len(slow.fruits) == len(normal.fruits)
    for a, b in zip(slow.fruits, normal.fruits):
        assert abs(a.x - b.x) < 1.0 and abs(a.y - b.y) < 1.0


def test_combo_banner_runs_in_real_time_during_slow_motion():
    g = Game(W, H, rng=random.Random(0))
    g.time_scale = 0.5
    g.combo_text, g.combo_banner_left = "COMBO x3!", config.COMBO_BANNER_SECONDS
    g.update(Blade(), 0.2)                                   # 0.2 real s, 0.1 game s
    assert g.combo_banner_left == pytest.approx(config.COMBO_BANNER_SECONDS - 0.2, abs=1e-6)


def test_long_pause_is_capped():
    g = Game(W, H, rng=random.Random(0))
    f = _fruit_at(W / 2, H / 2)
    f.vx, f.vy = 100.0, 0.0
    g.fruits = [f]
    g.update(Blade(), 5.0)                                   # e.g. camera switch blocked
    assert f.x == pytest.approx(W / 2 + 100.0 * config.MAX_FRAME_DT)


def test_default_step_is_one_reference_frame():
    g = Game(W, H, rng=random.Random(0))
    f = _fruit_at(W / 2, H / 2)
    f.vx, f.vy = 30.0, 0.0
    g.fruits = [f]
    g.update(Blade())
    assert f.x == pytest.approx(W / 2 + 30.0 / config.REF_FPS)


# -- CR-12 (#64): a bomb in the same swipe ends the run, the fruits cut before it still count
class _SoundSpy:
    def __init__(self):
        self.played = []

    def play(self, name):
        self.played.append(name)


def test_bomb_in_the_same_swipe_keeps_the_combo_of_the_fruits_cut_before_it():
    spy = _SoundSpy()
    g = Game(W, H, sounds=spy, rng=random.Random(0))
    falling = _fruit_at(W / 2, H + 100)             # would leave the screen in this very frame
    falling.vy = 5.0
    g.fruits = [_fruit_at(100, H / 2), _fruit_at(200, H / 2), _fruit_at(300, H / 2),
                _fruit_at(400, H / 2, bomb=True), falling]
    b = Blade()
    b.add((50, int(H / 2)))
    b.add((450, int(H / 2)))
    g.update(b, 1 / 30)
    assert g.game_over
    assert g.score == 3 + 3 * config.COMBO_BONUS_PER_FRUIT
    assert g.lives == g.difficulty.lives            # no physics step after the bomb
    assert "miss" not in spy.played
