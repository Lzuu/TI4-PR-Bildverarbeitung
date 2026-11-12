"""Integration tests through the real game loop (headless, fake camera) -- #41."""

import numpy as np

import main as m
from loop_harness import FakeCamera, green_frame, run_loop, with_pen


def _record_dress(monkeypatch):
    """Record the pen mask ('keep') the suit receives in every frame."""
    calls = []
    original = m.suit.dress

    def recording(frame, mask, face=None, keep=None, phase=0.0, smoother=None, **kwargs):
        calls.append(None if keep is None else keep.copy())
        return original(frame, mask, face, keep, phase, smoother, **kwargs)

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


def test_suit_receives_pristine_current_camera_image(monkeypatch, tmp_path):
    seen=[]
    original=m.suit.dress
    def recording(frame, mask, *args, **kwargs):
        seen.append(kwargs['original'].copy())
        return original(frame,mask,*args,**kwargs)
    monkeypatch.setattr(m.suit,'dress',recording)
    frames=[with_pen(green_frame(),75),with_pen(green_frame(),210)]
    run_loop(monkeypatch,tmp_path,FakeCamera(frames),2,BG_MODE='greenscreen',SUIT_ENABLED=True)
    assert np.array_equal(seen[0],frames[0])
    assert np.array_equal(seen[1],frames[1])


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

    def recording(frame, mask, face=None, keep=None, phase=0.0, smoother=None, **kw):
        faces.append(face)
        return original(frame, mask, face, keep, phase, smoother, **kw)
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


# -- #46: the saber hilt sits on the glove the suit drew in this frame -------
import types

import lightsaber

GLOVE = dict(name="glove_r", asset="glove",                     # wrist -> fingers, near the pen
             endpoints=np.array(((90, 330), (90, 430)), np.float32))


def _record_saber(monkeypatch):
    """Record every saber drawn and every skin-mask computation."""
    sabers, skins = [], []
    draw, skin_mask = lightsaber.draw_saber, lightsaber.skin_mask

    def recording_draw(frame, saber, color):
        sabers.append(saber)
        return draw(frame, saber, color)

    def recording_skin(frame):
        skins.append(1)
        return skin_mask(frame)
    monkeypatch.setattr(lightsaber, "draw_saber", recording_draw)
    monkeypatch.setattr(lightsaber, "skin_mask", recording_skin)
    return sabers, skins


def _hilt_offset(saber, glove):
    """Distance between the drawn hilt centre and the glove's grip point."""
    a, b = glove["endpoints"]
    grip = a + m.config.SABER_GLOVE_GRIP * (b - a)
    centre = (np.array(saber.hilt_top) + np.array(saber.hilt_bottom)) / 2
    return float(np.hypot(*(centre - grip)))


def test_saber_hilt_sits_on_the_glove_drawn_in_the_same_frame(monkeypatch, tmp_path):
    sabers, skins = _record_saber(monkeypatch)
    original = m.suit.dress

    def dress_with_glove(frame, mask, *args, **kwargs):
        out = original(frame, mask, *args, **kwargs)
        kwargs["renderer"].last_parts.append(dict(GLOVE))     # the suit drew a right glove
        return out
    monkeypatch.setattr(m.suit, "dress", dress_with_glove)
    run_loop(monkeypatch, tmp_path, FakeCamera([with_pen(green_frame(), 75)]), 3,
             BG_MODE="greenscreen", SUIT_ENABLED=True)
    drawn = [s for s in sabers if s is not None]
    assert len(drawn) == 3
    assert all(_hilt_offset(s, GLOVE) <= 1.5 for s in drawn)  # from the first frame on
    assert skins == []                                       # no skin search with a glove


def test_a_glove_from_the_previous_frame_is_not_reused(monkeypatch, tmp_path):
    """Frame 1: suit with glove. Frame 2: no player mask, so no suit -- skin fallback."""
    sabers, skins = _record_saber(monkeypatch)
    original_dress, original_mask = m.suit.dress, m.background.greenscreen_mask
    frames_seen = {"n": 0}

    def mask_only_in_the_first_frame(*args, **kwargs):
        frames_seen["n"] += 1
        return original_mask(*args, **kwargs) if frames_seen["n"] == 1 else None

    def dress_with_glove(frame, mask, *args, **kwargs):
        out = original_dress(frame, mask, *args, **kwargs)
        kwargs["renderer"].last_parts.append(dict(GLOVE))
        return out
    monkeypatch.setattr(m.background, "greenscreen_mask", mask_only_in_the_first_frame)
    monkeypatch.setattr(m.suit, "dress", dress_with_glove)
    run_loop(monkeypatch, tmp_path, FakeCamera([with_pen(green_frame(), 75)]), 2,
             BG_MODE="greenscreen", SUIT_ENABLED=True)
    first, _second = [s for s in sabers if s is not None]
    assert _hilt_offset(first, GLOVE) <= 1.5                 # frame 1: on the drawn glove
    assert skins == [1]                                      # frame 2: old glove ignored


def test_a_glove_from_an_earlier_frame_is_never_used(monkeypatch, tmp_path):
    sabers, skins = _record_saber(monkeypatch)
    stale = types.SimpleNamespace(last_parts=[dict(GLOVE)])   # drawn long ago, not this frame
    monkeypatch.setattr(m, "SuitRenderer", lambda: stale)
    run_loop(monkeypatch, tmp_path, FakeCamera([with_pen(green_frame(), 75)]), 3,
             BG_MODE="off", SUIT_ENABLED=True)
    drawn = [s for s in sabers if s is not None]
    assert drawn and skins                                   # "off": no suit, skin fallback
    assert all(_hilt_offset(s, GLOVE) > 10 for s in drawn)


# -- #33: challenge mode through the real loop ---------------------------------
from motion import LEFT, RIGHT
from challenge import ChallengeGame, DirectionFruit
from game import Game
import config as _config


class ScriptedCamera:
    """Frames are produced on demand: a pen position (screen coordinates) per read.

    The camera image is mirrored here when ``mirror`` is on, because the loop mirrors
    it again -- so the pen shows up at ``pen()`` on the screen the player sees.
    """

    index = 0

    def __init__(self, pen, mirror):
        self.pen = pen
        self.mirror = mirror
        self.reads = 0

    def read(self):
        self.reads += 1
        pos = self.pen(self.reads)
        frame = green_frame(person=False)
        if pos is not None:
            x, y = pos
            cam_x = _config.WIDTH - x if self.mirror else x
            frame = with_pen(frame, int(cam_x), int(y) + 30)     # tip = upper end of the pen
        return True, frame

    def switch(self, i):
        return True

    def release(self):
        pass


def _play_through_menu(monkeypatch, tmp_path, mode, mirror=False, swipe=None, fruit=None,
                       frames=400):
    """Menu -> mode circle -> difficulty circle -> play; then ``swipe`` (screen points)."""
    w, h = _config.WIDTH, _config.HEIGHT
    mode_t = m.challenge_target(w, h) if mode == m.MODE_CHALLENGE else m.menu_start_target(w, h)
    diff_t = m.row_targets(w, h, len(m.LEVELS))[2]
    games, state = [], {"swipe_at": None}
    real_new_game = m.new_game

    def recording_new_game(*a, **k):
        g = real_new_game(*a, **k)
        games.append(g)
        return g
    monkeypatch.setattr(m, "new_game", recording_new_game)

    def pen(read):
        if state["swipe_at"] is not None:
            k = read - state["swipe_at"]
            if swipe and 0 <= k < len(swipe):
                return swipe[k]
            return (40, 40)                                    # parked, away from everything
        if games:
            return (40, 40)
        # alternate between the two circles until the game exists: robust against timing
        return (mode_t[0], mode_t[1]) if (read // 8) % 2 == 0 else (diff_t[0], diff_t[1])

    camera = ScriptedCamera(pen, mirror)

    def on_frame(n, mouse):
        if games and state["swipe_at"] is None:
            if fruit is not None:
                games[0].fruits = [fruit]
            state["swipe_at"] = camera.reads + 3              # a few parked frames first
        if state["swipe_at"] is not None:
            return camera.reads > state["swipe_at"] + (len(swipe) if swipe else 0) + 3
        return False

    run_loop(monkeypatch, tmp_path, camera, frames, on_frame=on_frame, BG_MODE="off",
             SABER_ENABLED=False, DWELL_SECONDS=0.001, MIRROR=mirror, CUT_MIN_SPEED=0.02)
    return games


def _parked_fruit(direction, x, y):
    f = DirectionFruit(_config.WIDTH, _config.HEIGHT, direction)
    f.x, f.y, f.radius = float(x), float(y), 45
    f.vx = f.vy = f.gravity = 0.0
    return f


def _swipe_screen(x0, x1, y, step=60):
    n = int(abs(x1 - x0) / step) + 1
    return [(x0 + (x1 - x0) * k / (n - 1), y) for k in range(n)]


def test_challenge_selected_in_the_menu_starts_a_challenge(monkeypatch, tmp_path):
    games = _play_through_menu(monkeypatch, tmp_path, m.MODE_CHALLENGE)
    assert len(games) == 1 and isinstance(games[0], ChallengeGame)


def test_start_selected_in_the_menu_still_starts_the_classic_game(monkeypatch, tmp_path):
    games = _play_through_menu(monkeypatch, tmp_path, m.MODE_CLASSIC)
    assert len(games) == 1 and type(games[0]) is Game


def test_camera_motion_cuts_a_direction_fruit(monkeypatch, tmp_path):
    w, h = _config.WIDTH, _config.HEIGHT
    fruit = _parked_fruit(RIGHT, w / 2, h * 0.7)
    games = _play_through_menu(monkeypatch, tmp_path, m.MODE_CHALLENGE, fruit=fruit,
                               swipe=_swipe_screen(w * 0.2, w * 0.8, h * 0.7))
    assert fruit.sliced and games[0].directions_done == 1


def test_wrong_camera_direction_does_not_cut(monkeypatch, tmp_path):
    w, h = _config.WIDTH, _config.HEIGHT
    fruit = _parked_fruit(RIGHT, w / 2, h * 0.7)
    games = _play_through_menu(monkeypatch, tmp_path, m.MODE_CHALLENGE, fruit=fruit,
                               swipe=_swipe_screen(w * 0.8, w * 0.2, h * 0.7))
    assert not fruit.sliced and games[0].wrong_cuts >= 1


def test_mirrored_camera_image_and_arrow_use_the_same_screen_direction(monkeypatch, tmp_path):
    # The player sees the mirrored picture: the arrow refers to the screen. A pen moving
    # to the screen's left must cut a LEFT fruit although the raw camera image shows it
    # moving to the right.
    w, h = _config.WIDTH, _config.HEIGHT
    left_fruit = _parked_fruit(LEFT, w / 2, h * 0.7)
    games = _play_through_menu(monkeypatch, tmp_path, m.MODE_CHALLENGE, mirror=True,
                               fruit=left_fruit,
                               swipe=_swipe_screen(w * 0.8, w * 0.2, h * 0.7))
    assert left_fruit.sliced and games[0].directions_done == 1
