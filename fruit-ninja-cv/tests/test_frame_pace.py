"""Age budgets that follow a low frame rate (#88) -- pure, no camera and no models."""

import pytest

import config
from frame_pace import FramePace


def _tick(pace, interval, n, start=10.0):
    t = start
    pace.tick(t)
    for _ in range(n):
        t += interval
        pace.tick(t)
    return t


def test_budget_is_the_base_until_enough_frames_were_seen():
    pace = FramePace()
    _tick(pace, 0.3, 3)                               # only 3 intervals: still warming up
    assert pace.interval is None
    assert pace.budget(0.2, 3, 1.0) == 0.2


def test_a_steadily_slow_frame_rate_widens_the_budget():
    pace = FramePace()
    _tick(pace, 0.3, 8)
    assert pace.interval == pytest.approx(0.3)
    assert pace.budget(0.2, 3, 1.0) == pytest.approx(0.9)    # three frames


def test_budget_never_drops_below_the_base_at_a_normal_frame_rate():
    pace = FramePace()
    _tick(pace, 1 / 30, 20)
    assert pace.budget(0.2, 3, 1.0) == 0.2                   # 3 frames = 0.1 s < base


def test_budget_is_capped():
    pace = FramePace()
    _tick(pace, 5.0, 8)
    assert pace.budget(0.2, 3, 1.0) == 1.0


def test_the_cap_never_undercuts_the_base():
    pace = FramePace()
    _tick(pace, 0.3, 8)
    assert pace.budget(1.5, 3, 1.0) == 1.5


def test_a_single_long_frame_does_not_change_the_budget():
    # a hiccup is not a frame rate: the median of the recent intervals ignores it
    pace = FramePace()
    t = _tick(pace, 1 / 30, 7)
    pace.tick(t + 0.6)
    assert pace.interval == pytest.approx(1 / 30)
    assert pace.budget(0.2, 3, 1.0) == 0.2


def test_the_budget_follows_a_lasting_change():
    pace = FramePace()
    t = _tick(pace, 1 / 30, 8)
    _tick(pace, 0.3, 8, start=t)                             # the game became slow and stays slow
    assert pace.budget(0.2, 3, 1.0) == pytest.approx(0.9)


def test_repeated_or_backward_times_are_ignored():
    pace = FramePace()
    t = _tick(pace, 0.3, 6)
    pace.tick(t)                                             # same time again
    pace.tick(t - 5)                                         # clock going backwards
    assert pace.interval == pytest.approx(0.3)


def test_shared_constants_exist():
    assert config.AGE_BUDGET_FRAMES >= 2 and config.AGE_BUDGET_CAP >= config.POSE_MAX_AGE
