"""Challenge mode: direction fruits and camera-based cut detection (#33) -- headless."""

import random

import pytest

import config
from blade import Blade
from challenge import ChallengeGame, DirectionFruit
from fruit import Fruit
from motion import DIRECTIONS, DOWN, LEFT, RIGHT, UP, StrokeTracker

W, H = config.WIDTH, config.HEIGHT
FPS = 30
FAST = 3.0 * W / FPS          # px per frame = 3 field widths per second at 30 fps


def _game(seed=1, **kw):
    return ChallengeGame(W, H, rng=random.Random(seed), **kw)


def _dfruit(x, y, direction, r=45, bomb=False):
    f = DirectionFruit(W, H, direction)
    f.is_bomb = bomb
    f.x, f.y, f.radius = float(x), float(y), r
    f.vx = f.vy = f.gravity = 0.0
    return f


def _plain(x, y, r=45, bomb=False):
    f = Fruit(W, H)
    f.is_bomb = bomb
    f.x, f.y, f.radius = float(x), float(y), r
    f.vx = f.vy = f.gravity = 0.0
    return f


class Rig:
    """A game plus the blade and the stroke tracker, fed like the real loop does."""

    def __init__(self, game=None, fps=FPS):
        self.game = game or _game()
        self.blade = Blade()
        self.motion = StrokeTracker(W)
        self.fps = fps
        self.t = 100.0

    def step(self, point):
        self.blade.add(point)
        self.motion.add(point, self.t)
        self.game.update(self.blade, 1.0 / self.fps, self.motion)
        self.t += 1.0 / self.fps

    def run(self, points):
        for p in points:
            self.step(p)


def _line(a, b, n):
    return [(a[0] + (b[0] - a[0]) * k / (n - 1), a[1] + (b[1] - a[1]) * k / (n - 1))
            for k in range(n)]


def _swipe(direction, through, speed_px=FAST, n=8):
    """Points moving in ``direction`` that pass through ``through`` about halfway."""
    half = speed_px * (n - 1) / 2
    start = (through[0] - direction.vx * half, through[1] - direction.vy * half)
    end = (through[0] + direction.vx * half, through[1] + direction.vy * half)
    return _line(start, end, n)


# -- the fruits -----------------------------------------------------------------
def test_direction_fruit_carries_a_direction():
    f = DirectionFruit(W, H, UP)
    assert f.direction is UP and not f.is_bomb


def test_all_four_directions_are_thrown():
    g = _game(seed=3)
    seen = {f.direction for f in (g._new_fruit() for _ in range(400)) if getattr(f, "direction", None)}
    assert seen == set(DIRECTIONS)


def test_directions_are_spread_evenly():
    g = _game(seed=5)
    counts = {d: 0 for d in DIRECTIONS}
    for _ in range(800):
        f = g._new_fruit()
        if getattr(f, "direction", None):
            counts[f.direction] += 1
    total = sum(counts.values())
    assert total > 300
    assert all(abs(c / total - 0.25) < 0.05 for c in counts.values())


def test_bombs_and_plain_fruits_are_mixed_in():
    g = _game(seed=7)
    fruits = [g._new_fruit() for _ in range(600)]
    bombs = [f for f in fruits if f.is_bomb]
    directional = [f for f in fruits if getattr(f, "direction", None)]
    plain = [f for f in fruits if not f.is_bomb and not getattr(f, "direction", None)]
    assert bombs and plain and directional
    assert all(getattr(b, "direction", None) is None for b in bombs)
    share = len(directional) / (len(directional) + len(plain))
    assert abs(share - config.CHALLENGE_DIRECTION_SHARE) < 0.07


def test_spawning_over_time_produces_direction_fruits():
    g = _game(seed=2)
    g.lives = 10 ** 6                                   # keep playing while fruits fall past
    rig = Rig(g)
    seen = set()
    for _ in range(FPS * 40):
        rig.step(None)
        seen.update(f.direction for f in g.fruits if getattr(f, "direction", None))
    assert seen == set(DIRECTIONS)                      # thrown by the normal spawn timer


# -- valid cuts -------------------------------------------------------------------
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_fast_cut_in_the_arrow_direction_scores(direction):
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, direction)
    rig.game.fruits = [fruit]
    rig.run(_swipe(direction, (W / 2, H / 2)))
    assert fruit.sliced
    assert rig.game.score == config.CHALLENGE_POINTS
    assert rig.game.directions_done == 1
    assert [f.kind for f in rig.game.feedback] == ["good"]


def test_cut_between_two_frames_is_found_with_few_samples():
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, RIGHT, r=40)
    rig.game.fruits = [fruit]
    rig.run([(W / 2 - 200, H / 2), (W / 2 + 200, H / 2)])    # the pen jumps over the fruit
    assert fruit.sliced and rig.game.directions_done == 1


def test_cut_off_target_does_not_score():
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, RIGHT, r=40)
    rig.game.fruits = [fruit]
    rig.run(_swipe(RIGHT, (W / 2, H / 2 - 200)))              # fast and right, but far away
    assert not fruit.sliced and rig.game.score == 0 and rig.game.feedback == []


def test_a_fruit_is_scored_only_once():
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, RIGHT)
    rig.game.fruits = [fruit]
    for _ in range(3):
        rig.run(_swipe(RIGHT, (W / 2, H / 2)))
        rig.step(None)
    assert rig.game.score == config.CHALLENGE_POINTS
    assert rig.game.directions_done == 1


def test_two_fruits_in_one_swipe_each_score_once():
    rig = Rig()
    a, b = _dfruit(W / 2 - 60, H / 2, RIGHT, r=30), _dfruit(W / 2 + 60, H / 2, RIGHT, r=30)
    rig.game.fruits = [a, b]
    rig.run(_swipe(RIGHT, (W / 2, H / 2), n=10))
    assert a.sliced and b.sliced
    assert rig.game.directions_done == 2
    assert rig.game.score >= 2 * config.CHALLENGE_POINTS


# -- wrong direction ----------------------------------------------------------------
@pytest.mark.parametrize("direction,opposite", [(RIGHT, LEFT), (LEFT, RIGHT), (DOWN, UP), (UP, DOWN)])
def test_opposite_direction_does_not_cut_and_shows_negative_feedback(direction, opposite):
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, direction)
    rig.game.fruits = [fruit]
    rig.run(_swipe(opposite, (W / 2, H / 2)))
    assert not fruit.sliced
    assert rig.game.score == 0 and rig.game.directions_done == 0
    assert rig.game.lives == rig.game.difficulty.lives          # no life lost for a wrong cut
    assert "wrong" in [f.kind for f in rig.game.feedback]
    assert rig.game.wrong_cuts >= 1


def test_wrong_cut_feedback_is_brief():
    rig = Rig()
    rig.game.fruits = [_dfruit(W / 2, H / 2, RIGHT)]
    rig.run(_swipe(LEFT, (W / 2, H / 2)))
    assert rig.game.feedback
    for _ in range(int(FPS * (config.CHALLENGE_FEEDBACK_SECONDS + 0.5))):
        rig.step(None)
    assert rig.game.feedback == []


def test_wrong_cut_feedback_is_not_repeated_every_frame():
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, RIGHT, r=200)               # the pen stays inside the fruit
    rig.game.fruits = [fruit]
    rig.run(_swipe(LEFT, (W / 2, H / 2), n=12))
    assert rig.game.wrong_cuts == 1


def test_a_correct_cut_still_works_after_a_wrong_one():
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, RIGHT)
    rig.game.fruits = [fruit]
    rig.run(_swipe(LEFT, (W / 2, H / 2)))
    rig.step(None)
    assert not fruit.sliced
    rig.run(_swipe(RIGHT, (W / 2, H / 2)))
    assert fruit.sliced and rig.game.directions_done == 1


# -- slow / still / reacquired -------------------------------------------------------
def test_slow_touch_does_not_cut():
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, RIGHT)
    rig.game.fruits = [fruit]
    slow_px = 0.1 * W / FPS                                       # 0.1 field widths per second
    rig.run(_swipe(RIGHT, (W / 2, H / 2), speed_px=slow_px, n=12))
    assert not fruit.sliced and rig.game.score == 0
    assert rig.game.feedback == []                                 # a gentle touch is not an error


def test_standing_still_on_a_fruit_does_not_cut():
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, RIGHT)
    rig.game.fruits = [fruit]
    rig.run([(W / 2, H / 2)] * 30)
    assert not fruit.sliced and rig.game.score == 0 and rig.game.feedback == []


def test_a_moderately_slow_swipe_gets_a_hint_but_no_cut():
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, RIGHT)
    rig.game.fruits = [fruit]
    px = 0.6 * config.CUT_MIN_SPEED * W / FPS
    rig.run(_swipe(RIGHT, (W / 2, H / 2), speed_px=px, n=10))
    assert not fruit.sliced
    assert [f.kind for f in rig.game.feedback] == ["slow"]


def test_reappearing_pen_does_not_cut_across_the_field():
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, RIGHT, r=40)
    rig.game.fruits = [fruit]
    rig.run(_swipe(RIGHT, (80, H / 2), n=5))                     # stroke at the left edge
    rig.step(None)                                               # tracking lost
    rig.run([(W - 80, H / 2), (W - 40, H / 2)])                  # found again far to the right
    assert not fruit.sliced and rig.game.score == 0


def test_cutting_works_again_after_a_tracking_gap():
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, RIGHT)
    rig.game.fruits = [fruit]
    rig.run(_swipe(RIGHT, (80, 80), n=4))
    rig.step(None)
    rig.run(_swipe(RIGHT, (W / 2, H / 2)))
    assert fruit.sliced


# -- classic fruits and bombs keep their rules --------------------------------------
def test_plain_fruit_is_cut_by_touch_like_in_the_classic_game():
    rig = Rig()
    fruit = _plain(W / 2, H / 2)
    rig.game.fruits = [fruit]
    rig.step((W / 2, H / 2))
    assert fruit.sliced and rig.game.score == 1 and rig.game.directions_done == 0


def test_bomb_ends_the_challenge():
    rig = Rig()
    rig.game.fruits = [_plain(W / 2, H / 2, bomb=True)]
    rig.step((W / 2, H / 2))
    assert rig.game.game_over and not rig.game.time_up


def test_missed_direction_fruit_costs_a_life():
    rig = Rig()
    f = _dfruit(W / 2, H + 100, RIGHT)
    f.vy = 50.0
    rig.game.fruits = [f]
    rig.step(None)
    assert rig.game.lives == rig.game.difficulty.lives - 1


def test_a_missed_bomb_is_free():
    rig = Rig()
    b = _plain(W / 2, H + 100, bomb=True)
    b.vy = 50.0
    rig.game.fruits = [b]
    rig.step(None)
    assert rig.game.lives == rig.game.difficulty.lives


# -- timer, lives, summary ---------------------------------------------------------
def test_challenge_lasts_sixty_seconds_by_default():
    assert _game().time_left == config.CHALLENGE_SECONDS == 60


def test_timer_counts_down_in_real_seconds():
    g = _game()
    for _ in range(20):
        g.update(Blade(), 0.25)
    assert g.time_left == pytest.approx(60 - 5.0)


def test_challenge_ends_when_the_time_is_up():
    g = _game(seconds=2.0)
    g.lives = 10 ** 6
    for _ in range(100):
        g.update(Blade(), 0.1)
        if g.game_over:
            break
    assert g.game_over and g.time_up and g.time_left == 0


def test_timer_ignores_slow_motion():
    g = _game()
    g.time_scale = 0.5
    g.update(Blade(), 0.2)
    assert g.time_left == pytest.approx(60 - 0.2)


def test_game_over_by_lives_is_not_time_up():
    g = _game()
    g.lives = 1
    f = _dfruit(W / 2, H + 100, RIGHT)
    f.vy = 50.0
    g.fruits = [f]
    g.update(Blade(), 0.03)
    assert g.game_over and not g.time_up and g.time_left > 50


def test_finished_game_is_frozen():
    g = _game(seconds=0.1)
    g.update(Blade(), 0.2)
    assert g.game_over
    left, score = g.time_left, g.score
    g.update(Blade(), 0.2)
    assert (g.time_left, g.score) == (left, score)


def test_reset_restores_timer_and_counters():
    rig = Rig()
    rig.game.fruits = [_dfruit(W / 2, H / 2, RIGHT)]
    rig.run(_swipe(RIGHT, (W / 2, H / 2)))
    assert rig.game.directions_done == 1
    rig.game.reset()
    g = rig.game
    assert (g.score, g.directions_done, g.wrong_cuts) == (0, 0, 0)
    assert g.time_left == config.CHALLENGE_SECONDS and not g.game_over and not g.time_up
    assert g.feedback == [] and g.fruits == []


def test_leaderboard_name_separates_the_modes():
    g = _game()
    assert "Challenge" in g.board_name and g.difficulty.name in g.board_name


def test_decisions_are_independent_of_the_frame_rate():
    def outcome(fps, direction):
        rig = Rig(fps=fps)
        fruit = _dfruit(W / 2, H / 2, RIGHT)
        rig.game.fruits = [fruit]
        n = max(3, int(round(fps * 0.25)))
        step = 2.4 * W / fps                                    # 2.4 field widths per second
        rig.run(_swipe(direction, (W / 2, H / 2), speed_px=step, n=n))
        return fruit.sliced, rig.game.wrong_cuts

    for fps in (10, 15, 30, 60):
        assert outcome(fps, RIGHT) == (True, 0), fps
        assert outcome(fps, LEFT) == (False, 1), fps


def test_drawing_does_not_crash():
    import numpy as np
    rig = Rig()
    rig.game.fruits = [_dfruit(300, 200, d) for d in DIRECTIONS]
    rig.game.feedback_add("good", "+3", (300, 200))
    rig.game.feedback_add("wrong", "Falsche Richtung", (300, 200))
    frame = np.zeros((H, W, 3), np.uint8)
    rig.game.draw(frame)
    rig.game.draw_hud(frame)
    rig.game.draw_feedback(frame)
    rig.game.draw_combo(frame)
    assert frame.any()


def test_an_impossible_jump_over_a_fruit_is_not_a_cut():
    # e.g. the tracker latching onto another pink object for one frame
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, RIGHT, r=40)
    rig.game.fruits = [fruit]
    rig.run([(20.0, H / 2), (W - 20.0, H / 2)])
    assert not fruit.sliced and rig.game.score == 0 and rig.game.feedback == []


def test_cutting_works_right_after_an_impossible_jump():
    rig = Rig()
    fruit = _dfruit(W / 2, H / 2, LEFT)
    rig.game.fruits = [fruit]
    rig.run([(20.0, 20.0), (W - 100.0, H / 2)])          # jump, then a swipe to the left
    rig.run(_swipe(LEFT, (W / 2, H / 2)))
    assert fruit.sliced and rig.game.directions_done == 1
