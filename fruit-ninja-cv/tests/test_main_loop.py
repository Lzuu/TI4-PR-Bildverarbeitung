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
