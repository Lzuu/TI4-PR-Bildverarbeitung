"""Integration tests through the real game loop (headless, fake camera) -- #41."""

import numpy as np

import main as m
from loop_harness import FakeCamera, green_frame, run_loop, with_pen


def _record_dress(monkeypatch):
    """Record the pen mask ('keep') the suit receives in every frame."""
    calls = []
    original = m.suit.dress

    def recording(frame, mask, face=None, keep=None, phase=0.0, smoother=None):
        calls.append(None if keep is None else keep.copy())
        return original(frame, mask, face, keep, phase, smoother)

    monkeypatch.setattr(m.suit, "dress", recording)
    return calls


# -- CR-06: the suit gets the pen mask of the *current* frame ----------------
def test_suit_gets_the_pen_mask_of_the_current_frame(monkeypatch, tmp_path):
    calls = _record_dress(monkeypatch)
    frames = [with_pen(green_frame(), 75), with_pen(green_frame(), 210)]
    run_loop(monkeypatch, tmp_path, FakeCamera(frames), 2,
             BG_MODE="greenscreen", SUIT_ENABLED=True)
    assert len(calls) == 2
    first, second = calls
    assert first is not None and first[300, 75] > 0         # first frame already protected
    assert second[300, 210] > 0                             # new pen position protected
    assert second[300, 75] == 0                             # not the old one


# -- CR-09 / CR-10: YOLO in the real loop ------------------------------------
import cv2

import yolo_seg
from test_yolo_seg import FakeNet, FailingNet, _blank_outputs


def _use_segmenter(monkeypatch, net):
    seg = yolo_seg.BodySegmenter(net, input_size=64, every_n=100)
    monkeypatch.setattr(yolo_seg, "create_segmenter", lambda: seg)
    return seg


def test_yolo_inference_error_does_not_end_the_game(monkeypatch, tmp_path):
    net = FailingNet(*_blank_outputs())
    _use_segmenter(monkeypatch, net)
    frames_run = run_loop(monkeypatch, tmp_path, FakeCamera([green_frame()]), 6,
                          BG_MODE="yolo", YOLO_ENABLED=True)
    assert frames_run == 6                                 # loop kept running until 'q'
    assert net.calls == 1                                  # disabled, no retry per frame


def test_camera_switch_triggers_a_fresh_yolo_mask(monkeypatch, tmp_path):
    net = FakeNet(*_blank_outputs())
    _use_segmenter(monkeypatch, net)
    camera = FakeCamera([green_frame()])

    def switch_camera(n, mouse):
        if n == 2:
            camera.index = 1                               # successful switch, same size
    run_loop(monkeypatch, tmp_path, camera, 4, on_frame=switch_camera,
             BG_MODE="yolo", YOLO_ENABLED=True)
    assert net.calls == 2                                  # first frame + new camera


def test_returning_to_yolo_does_not_reuse_an_old_mask(monkeypatch, tmp_path):
    net = FakeNet(*_blank_outputs())
    _use_segmenter(monkeypatch, net)
    bx, by, bw, bh = m.background_button(m.config.WIDTH, m.config.HEIGHT)

    def click_mode(n, mouse):
        if 2 <= n <= 5:                                    # yolo -> off -> greenscreen -> normal -> yolo
            mouse(cv2.EVENT_LBUTTONDOWN, bx + 5, by + 5, 0, None)
    run_loop(monkeypatch, tmp_path, FakeCamera([green_frame()]), 8, on_frame=click_mode,
             BG_MODE="yolo", YOLO_ENABLED=True)
    assert net.calls == 2                                  # once before, once after returning


# -- CR-04: menu clicks keep UI and tracker consistent -----------------------
from colors import COLORS, GRUEN, PINK


def _record_tracker_colors(monkeypatch):
    colors = []
    original = m.MarkerTracker.set_color

    def recording(self, pen):
        colors.append(pen)
        return original(self, pen)
    monkeypatch.setattr(m.MarkerTracker, "set_color", recording)
    return colors


def _click(mouse, rect_or_circle):
    if len(rect_or_circle) == 4:
        x, y, w, h = rect_or_circle
        mouse(cv2.EVENT_LBUTTONDOWN, x + 5, y + 5, 0, None)
    else:
        cx, cy, r = rect_or_circle
        mouse(cv2.EVENT_LBUTTONDOWN, cx, cy, 0, None)


def test_green_cannot_be_picked_in_greenscreen_mode(monkeypatch, tmp_path):
    colors = _record_tracker_colors(monkeypatch)
    W, H = m.config.WIDTH, m.config.HEIGHT
    green = m.color_swatches(W, H)[COLORS.index(GRUEN)]

    def clicks(n, mouse):
        if n == 1:
            _click(mouse, m.settings_button(W, H))       # open the colour panel
        elif n == 2:
            _click(mouse, green)                         # try green
    run_loop(monkeypatch, tmp_path, FakeCamera([green_frame()]), 4, on_frame=clicks,
             BG_MODE="greenscreen")
    assert GRUEN not in colors and colors[-1] is PINK


def test_switching_to_greenscreen_with_a_green_pen_falls_back_to_pink(monkeypatch, tmp_path):
    colors = _record_tracker_colors(monkeypatch)
    W, H = m.config.WIDTH, m.config.HEIGHT
    green = m.color_swatches(W, H)[COLORS.index(GRUEN)]

    def clicks(n, mouse):
        if n == 1:
            _click(mouse, m.settings_button(W, H))
        elif n == 2:
            _click(mouse, green)                         # allowed in normal mode
        elif 3 <= n <= 5:                                # normal -> yolo -> off -> greenscreen
            _click(mouse, m.background_button(W, H))
    run_loop(monkeypatch, tmp_path, FakeCamera([green_frame()]), 8, on_frame=clicks,
             BG_MODE="normal")
    assert GRUEN in colors                               # green was picked first
    assert colors[-1] is PINK                            # and replaced for greenscreen


# -- PERF-07 (#42): YOLO is only loaded when the yolo mode is used ------------
import pytest


def _counting_create(monkeypatch, net):
    calls = []

    def create():
        calls.append(1)
        return yolo_seg.BodySegmenter(net, input_size=64, every_n=100)
    monkeypatch.setattr(yolo_seg, "create_segmenter", create)
    return calls


@pytest.mark.parametrize("mode", ["off", "normal", "greenscreen"])
def test_other_modes_start_without_loading_yolo(monkeypatch, tmp_path, mode):
    calls = _counting_create(monkeypatch, FakeNet(*_blank_outputs()))
    run_loop(monkeypatch, tmp_path, FakeCamera([green_frame()]), 4,
             BG_MODE=mode, YOLO_ENABLED=True)
    assert calls == []


def test_yolo_is_loaded_once_when_first_selected(monkeypatch, tmp_path):
    calls = _counting_create(monkeypatch, FakeNet(*_blank_outputs()))
    bx, by, bw, bh = m.background_button(m.config.WIDTH, m.config.HEIGHT)

    def click_mode(n, mouse):
        if 2 <= n <= 9:                              # normal -> yolo -> ... -> yolo again
            mouse(cv2.EVENT_LBUTTONDOWN, bx + 5, by + 5, 0, None)
    run_loop(monkeypatch, tmp_path, FakeCamera([green_frame()]), 12, on_frame=click_mode,
             BG_MODE="normal", YOLO_ENABLED=True)
    assert calls == [1]                              # loaded on first use, then reused


def test_loading_hint_while_the_model_loads(monkeypatch, tmp_path):
    _counting_create(monkeypatch, FakeNet(*_blank_outputs()))
    jobs = []
    texts = []
    original = m.draw_text

    def recording(frame, text, *a, **k):
        texts.append(text)
        return original(frame, text, *a, **k)
    monkeypatch.setattr(m, "draw_text", recording)
    run_loop(monkeypatch, tmp_path, FakeCamera([green_frame()]), 3,
             BG_MODE="yolo", YOLO_ENABLED=True, yolo_start=jobs.append)
    assert len(jobs) == 1                            # one background load, never finished
    assert any("wird geladen" in t for t in texts)


# -- PERF-05 (#42): one face detection per frame, shared by suit, saber and scan
class CountingIdentifier:
    """Stands in for FaceIdentifier; counts how often a detection is run."""

    persons = []
    BOX = (120, 90, 80, 80)

    def __init__(self):
        self.scan_calls = 0
        self.box_calls = 0

    def scan(self, frame):
        self.scan_calls += 1
        return None, 0.0, self.BOX, None

    def largest_face_box(self, frame, width=320):
        self.box_calls += 1
        return self.BOX


def _scan_loop(monkeypatch, tmp_path, state, frames=4, **cfg):
    ident = CountingIdentifier()
    monkeypatch.setattr(m, "initial_state", lambda identifier: state)
    dress = _record_face_args(monkeypatch)
    run_loop(monkeypatch, tmp_path, FakeCamera([with_pen(green_frame(), 200)]), frames,
             identifier=ident, BG_MODE="greenscreen", SUIT_ENABLED=True,
             SABER_ENABLED=True, **cfg)
    return ident, dress


def _record_face_args(monkeypatch):
    faces = []
    original = m.suit.dress

    def recording(frame, mask, face=None, keep=None, phase=0.0, smoother=None):
        faces.append(face)
        return original(frame, mask, face, keep, phase, smoother)
    monkeypatch.setattr(m.suit, "dress", recording)
    return faces


def test_face_scan_detects_once_per_frame_and_shares_the_box(monkeypatch, tmp_path):
    ident, faces = _scan_loop(monkeypatch, tmp_path, m.STATE_FACE_SCAN)
    assert ident.scan_calls == 4
    assert ident.box_calls == 0                      # no second detection for suit / saber
    assert faces == [CountingIdentifier.BOX] * 4     # helmet placed from the scan result


def test_menu_detects_once_per_frame_without_recognition(monkeypatch, tmp_path):
    ident, faces = _scan_loop(monkeypatch, tmp_path, m.STATE_MENU)
    assert ident.scan_calls == 0                     # no full recognition outside FACE_SCAN
    assert ident.box_calls == 4                      # suit and saber share one detection
    assert faces == [CountingIdentifier.BOX] * 4


def test_no_face_found_is_not_searched_again_for_the_saber(monkeypatch, tmp_path):
    ident = CountingIdentifier()
    ident.largest_face_box = lambda frame, width=320: setattr(
        ident, "box_calls", ident.box_calls + 1)     # finds nothing
    monkeypatch.setattr(m, "initial_state", lambda identifier: m.STATE_MENU)
    run_loop(monkeypatch, tmp_path, FakeCamera([with_pen(green_frame(), 200)]), 3,
             identifier=ident, BG_MODE="greenscreen", SUIT_ENABLED=True, SABER_ENABLED=True)
    assert ident.box_calls == 3                      # one try per frame, not two


# -- PERF-08 (#42): YOLO inference and mask age show up in the diagnostics ----
def test_yolo_inference_and_mask_age_are_reported(monkeypatch, tmp_path, caplog):
    import logging
    caplog.set_level(logging.INFO)
    _use_segmenter(monkeypatch, FakeNet(*_blank_outputs()))
    run_loop(monkeypatch, tmp_path, FakeCamera([green_frame()]), 6,
             BG_MODE="yolo", YOLO_ENABLED=True, PERF_LOG_INTERVAL=0.0)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("Perf ")]
    assert lines
    assert any("yolo_infer" in line and "max mask_age_ms=" in line for line in lines)
