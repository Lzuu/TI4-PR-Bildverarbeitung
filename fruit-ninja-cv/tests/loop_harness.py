"""Run the real game loop headless: fake camera, no window, scripted keys (#41).

Integration tests for things that only show up when several modules work
together in ``main._run_game_loop`` (frame order, camera/mode changes).
"""

import cv2
import numpy as np

import config
import main as m
from sound import NullSound


class FakeCamera:
    """Plays a list of frames; ``index`` and ``switch`` like cameras.Camera."""

    def __init__(self, frames, index=0, switch_ok=True):
        self.frames = list(frames)
        self.index = index
        self.switch_ok = switch_ok
        self.reads = 0

    def read(self):
        frame = self.frames[min(self.reads, len(self.frames) - 1)]
        self.reads += 1
        return True, frame.copy()

    def switch(self, index):
        if self.switch_ok:
            self.index = index
        return self.switch_ok

    def release(self):
        pass


def run_loop(monkeypatch, tmp_path, camera, frames_to_run, keys=None, on_frame=None,
             identifier=None, **cfg):
    """Run ``_run_game_loop`` for ``frames_to_run`` frames, then quit.

    ``keys``: {frame_number: keycode}; ``on_frame(n, mouse_cb)`` runs before each
    waitKey (e.g. to click); ``cfg`` overrides config values for this run.
    """
    settings = dict(SETTINGS_FILE=str(tmp_path / "settings.json"),
                    HIGHSCORE_FILE=str(tmp_path / "hs.json"),
                    YOLO_ENABLED=False, MIRROR=False, FULLSCREEN=False)
    settings.update(cfg)
    for name, value in settings.items():
        monkeypatch.setattr(config, name, value)
    keys = keys or {}
    state = {"n": 0, "mouse": None}

    def wait_key(_delay):
        state["n"] += 1
        if on_frame is not None:
            on_frame(state["n"], state["mouse"])
        if state["n"] >= frames_to_run:
            return ord("q")
        return keys.get(state["n"], 255)

    monkeypatch.setattr(cv2, "imshow", lambda *a: None)
    monkeypatch.setattr(cv2, "namedWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "setWindowProperty", lambda *a: None)
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a: None)
    monkeypatch.setattr(cv2, "setMouseCallback", lambda name, cb: state.__setitem__("mouse", cb))
    monkeypatch.setattr(cv2, "waitKey", wait_key)
    previews = {camera.index: np.zeros((108, 192, 3), np.uint8)}
    m._run_game_loop(camera, NullSound(), previews, identifier)
    return state["n"]


def green_frame(person=True):
    """Canvas-sized frame: green backdrop, optionally a grey 'person' block."""
    frame = np.full((config.HEIGHT, config.WIDTH, 3), (0, 200, 0), np.uint8)
    if person:
        frame[100:, 50:400] = (128, 128, 128)
    return frame


def with_pen(frame, x, y=300, color=(180, 0, 255)):
    """Draw a pink pen (20 x 60 px) centred at (x, y)."""
    out = frame.copy()
    cv2.rectangle(out, (x - 10, y - 30), (x + 10, y + 30), color, -1)
    return out
