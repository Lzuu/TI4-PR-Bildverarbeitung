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
