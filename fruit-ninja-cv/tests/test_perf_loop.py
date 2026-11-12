"""Frame-time diagnostics in the real game loop (#42, PERF-06) -- headless."""

import csv
import logging

import cv2

import main as m
from loop_harness import FakeCamera, green_frame, run_loop, with_pen


def _perf_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith("Perf ")]


def test_loop_logs_stage_times(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    frames = [with_pen(green_frame(), 200)]
    run_loop(monkeypatch, tmp_path, FakeCamera(frames), 5,
             BG_MODE="greenscreen", SUIT_ENABLED=True, PERF_LOG_INTERVAL=0.0)
    lines = _perf_lines(caplog)
    assert lines                                     # one summary per window
    last = lines[-1]
    for stage in ("read", "track", "segment", "suit", "screen", "saber", "show"):
        assert stage in last, (stage, last)


def test_failed_reads_are_counted(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    camera = FakeCamera([green_frame()], fail_reads={2, 3})
    run_loop(monkeypatch, tmp_path, camera, 8, BG_MODE="off", PERF_LOG_INTERVAL=0.0)
    assert any("read_fail=" in line for line in _perf_lines(caplog))


def test_no_perf_log_within_the_interval(monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO)
    run_loop(monkeypatch, tmp_path, FakeCamera([green_frame()]), 5,
             BG_MODE="off", PERF_LOG_INTERVAL=3600.0)
    assert _perf_lines(caplog) == []                 # not per frame


def test_perf_csv_export(monkeypatch, tmp_path):
    path = tmp_path / "perf.csv"
    run_loop(monkeypatch, tmp_path, FakeCamera([green_frame()]), 4,
             BG_MODE="off", PERF_LOG_INTERVAL=0.0, PERF_CSV=str(path))
    with path.open(encoding="utf-8", newline="") as f:
        stages = {r["stage"] for r in csv.DictReader(f)}
    assert {"frame", "read", "track"} <= stages


def test_key_p_toggles_the_overlay(monkeypatch, tmp_path):
    shown = []
    original = m.draw_perf_overlay

    def recording(frame, lines):
        shown.append(list(lines))
        return original(frame, lines)
    monkeypatch.setattr(m, "draw_perf_overlay", recording)
    run_loop(monkeypatch, tmp_path, FakeCamera([green_frame()]), 6,
             keys={2: ord("p"), 4: ord("p")}, BG_MODE="off", PERF_LOG_INTERVAL=0.0)
    assert len(shown) == 2                           # frames 3 and 4, then off again
    assert any("FPS" in line for line in shown[0])
