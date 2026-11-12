"""Frame-time diagnostics (#42, PERF-06) -- headless, with a fake clock."""

import csv

import pytest

import perf


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _frame(stats, clock, stages, gap=0.0):
    """One frame: each (name, seconds) stage advances the clock while timed."""
    stats.start_frame()
    for name, seconds in stages:
        with stats.stage(name):
            clock.t += seconds
    clock.t += gap
    return stats.end_frame()


def test_percentile_basics():
    assert perf.percentile([], 50) is None
    assert perf.percentile([5.0], 95) == 5.0
    values = list(range(1, 101))
    assert perf.percentile(values, 50) == pytest.approx(50.5)
    assert perf.percentile(values, 95) == pytest.approx(95.05)


def test_summary_after_interval_with_stage_stats():
    clock = Clock()
    stats = perf.FrameStats(interval=1.0, clock=clock, log=None)
    summary = None
    for _ in range(20):                     # 20 frames of 50 ms -> 1 s
        summary = summary or _frame(stats, clock, [("read", 0.01), ("track", 0.04)])
    assert summary is not None
    assert summary["frames"] == 20
    assert summary["fps"] == pytest.approx(20.0)
    assert summary["frame_ms"]["median"] == pytest.approx(50.0)
    assert summary["stages"]["track"]["median"] == pytest.approx(40.0)
    assert summary["stages"]["read"]["count"] == 20


def test_no_summary_before_interval():
    clock = Clock()
    stats = perf.FrameStats(interval=10.0, clock=clock, log=None)
    for _ in range(5):
        assert _frame(stats, clock, [("read", 0.03)]) is None


def test_stage_only_counted_in_frames_where_it_ran():
    # e.g. YOLO inference runs only every Nth frame: its stats must not be
    # diluted by the cache frames (#42, PERF-08)
    clock = Clock()
    stats = perf.FrameStats(interval=0.595, clock=clock, log=None)   # 30 frames = 0.6 s
    summary = None
    for i in range(30):
        stages = [("read", 0.01)] + ([("yolo_infer", 0.03)] if i % 3 == 0 else [])
        summary = summary or _frame(stats, clock, stages)
    assert summary["stages"]["yolo_infer"]["count"] == 10
    assert summary["stages"]["yolo_infer"]["median"] == pytest.approx(30.0)


def test_counters_and_maxima():
    clock = Clock()
    stats = perf.FrameStats(interval=1.0, clock=clock, log=None)
    stats.count("read_fail")
    stats.count("read_fail")
    stats.maximum("mask_age_ms", 40.0)
    stats.maximum("mask_age_ms", 90.0)
    stats.maximum("mask_age_ms", 10.0)
    summary = None
    while summary is None:
        summary = _frame(stats, clock, [("read", 0.1)])
    assert summary["counters"]["read_fail"] == 2
    assert summary["maxima"]["mask_age_ms"] == pytest.approx(90.0)


def test_window_resets_after_summary():
    clock = Clock()
    stats = perf.FrameStats(interval=1.0, clock=clock, log=None)
    first = None
    while first is None:
        first = _frame(stats, clock, [("read", 0.1)])
    stats.count("read_fail")
    second = None
    while second is None:
        second = _frame(stats, clock, [("read", 0.05)])
    assert second["frames"] == 20
    assert second["counters"]["read_fail"] == 1
    assert "read_fail" not in first["counters"]


def test_summary_is_logged_once_per_window():
    clock = Clock()
    lines = []
    stats = perf.FrameStats(interval=1.0, clock=clock, log=lambda *a: lines.append(a))
    for _ in range(25):
        _frame(stats, clock, [("read", 0.05)])
    assert len(lines) == 1                  # 25 frames, one 1-s window -> one line


def test_csv_export(tmp_path):
    path = tmp_path / "perf.csv"
    clock = Clock()
    stats = perf.FrameStats(interval=1.0, clock=clock, log=None, csv_path=str(path))
    for _ in range(45):                     # two full windows
        _frame(stats, clock, [("read", 0.01), ("track", 0.04)])
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    # long format: one row per window and stage (frame = whole frame)
    assert len(rows) == 2 * 3
    assert {r["stage"] for r in rows} == {"frame", "read", "track"}
    track = [r for r in rows if r["stage"] == "track"][0]
    assert float(track["fps"]) == pytest.approx(20.0)
    assert float(track["median_ms"]) == pytest.approx(40.0)


def test_overlay_lines_from_summary():
    clock = Clock()
    stats = perf.FrameStats(interval=1.0, clock=clock, log=None)
    assert stats.overlay_lines() == []      # nothing measured yet
    while _frame(stats, clock, [("track", 0.05)]) is None:
        pass
    lines = stats.overlay_lines()
    assert any("FPS" in line for line in lines)
    assert any("track" in line for line in lines)
