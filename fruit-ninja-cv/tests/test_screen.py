"""Screen detection and canvas size (#41, CR-08) -- headless, no real display."""

import subprocess
import types

import config
import main


class FakeUser32:
    def __init__(self, w, h):
        self.size = {0: w, 1: h}

    def GetSystemMetrics(self, index):
        return self.size[index]


def test_windows_uses_system_metrics(monkeypatch):
    user32 = FakeUser32(1440, 960)
    monkeypatch.setattr(main, "_windows_user32", lambda: user32)
    assert main.detect_screen_size("win32") == (1440, 960)


def test_windows_failure_falls_back(monkeypatch):
    def broken():
        raise OSError("no user32")
    monkeypatch.setattr(main, "_windows_user32", broken)
    assert main.detect_screen_size("win32") == main.DEFAULT_SCREEN


def test_windows_zero_metrics_fall_back(monkeypatch):
    monkeypatch.setattr(main, "_windows_user32", lambda: FakeUser32(0, 0))
    assert main.detect_screen_size("win32") == main.DEFAULT_SCREEN


def test_mac_reads_system_profiler(monkeypatch):
    out = "Resolution: 3024 x 1964 Retina\n  UI Looks like: 1512 x 982 @ 120.00Hz\n"
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: types.SimpleNamespace(stdout=out))
    assert main.detect_screen_size("darwin") == (1512, 982)


def test_unknown_platform_falls_back():
    assert main.detect_screen_size("linux") == main.DEFAULT_SCREEN


def test_windows_canvas_keeps_screen_aspect():
    # Target device: 1440x960 logical (3:2). Height capped for frame rate.
    w, h = main.canvas_size(1440, 960, "win32")
    assert h == config.MAX_CANVAS_HEIGHT
    assert abs(w / h - 1440 / 960) < 0.01


def test_windows_canvas_small_screen_not_upscaled():
    assert main.canvas_size(1280, 720, "win32") == (1280, 720)


def test_mac_canvas_unchanged():
    # macOS keeps the 16:10 usable fullscreen area and the full screen height.
    assert main.canvas_size(1512, 982, "darwin") == (round(982 * config.ASPECT), 982)
