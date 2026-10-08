"""Tests for the VR-style dwell selection (time-based, deterministic clock)."""

from dwell import DwellTracker


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


TARGETS = [(100, 100, 50), (300, 100, 50)]


def test_confirms_after_holding_full_duration():
    clk = FakeClock()
    d = DwellTracker(3.0, time_fn=clk)
    active, progress, confirmed = d.update((100, 100), TARGETS)
    assert active == 0 and confirmed is None and progress == 0.0
    clk.t = 1.5
    _, progress, confirmed = d.update((100, 100), TARGETS)
    assert 0.4 < progress < 0.6 and confirmed is None
    clk.t = 3.0
    _, progress, confirmed = d.update((100, 100), TARGETS)
    assert progress == 1.0 and confirmed == 0


def test_losing_the_pen_resets_progress():
    clk = FakeClock()
    d = DwellTracker(3.0, time_fn=clk)
    d.update((100, 100), TARGETS)
    clk.t = 2.0
    d.update((100, 100), TARGETS)
    clk.t = 2.1
    active, progress, confirmed = d.update(None, TARGETS)
    assert active is None and progress == 0.0 and confirmed is None
    clk.t = 2.2
    active, progress, _ = d.update((100, 100), TARGETS)
    assert active == 0 and progress == 0.0        # timer restarted


def test_switching_target_restarts_timer():
    clk = FakeClock()
    d = DwellTracker(3.0, time_fn=clk)
    d.update((100, 100), TARGETS)
    clk.t = 2.0
    d.update((100, 100), TARGETS)
    active, progress, _ = d.update((300, 100), TARGETS)   # moved to target 1
    assert active == 1 and progress == 0.0


def test_point_outside_all_targets():
    d = DwellTracker(3.0, time_fn=FakeClock())
    active, progress, confirmed = d.update((5, 5), TARGETS)
    assert active is None and progress == 0.0 and confirmed is None
