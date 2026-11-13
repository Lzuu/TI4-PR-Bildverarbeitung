"""Integration tests through the real game loop (headless, fake camera) -- #41."""

import numpy as np

import main as m
from loop_harness import FakeCamera, green_frame, run_loop, with_pen


def _record_dress(monkeypatch):
    """Record the pen mask ('keep') the suit receives in every frame."""
    calls = []
    original = m.suit.dress

    def recording(frame, mask, face=None, keep=None, phase=0.0, **kwargs):
        calls.append(None if keep is None else keep.copy())
        return original(frame, mask, face, keep, phase, **kwargs)

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

    def recording(frame, mask, face=None, keep=None, phase=0.0, **kw):
        faces.append(face)
        return original(frame, mask, face, keep, phase, **kw)
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


# -- CR-11 (#63): after a screen change the pen must leave the targets before a dwell counts
from dwell import DwellTracker
import face_id


def _fake_dwell_clock(monkeypatch, step=0.1):
    """Dwell timing on a fake clock that advances ``step`` seconds per frame."""
    clock = [0.0]
    monkeypatch.setattr(m, "DwellTracker",
                        lambda hold: DwellTracker(hold, time_fn=lambda: clock[0]))

    def tick(n, mouse):
        clock[0] += step
    return tick


def _pen_at_tip(x, y):
    """Frame with the pen tip (the upper end of the pen) at (x, y)."""
    return with_pen(green_frame(), x, y + 30)


def test_holding_still_after_start_does_not_pick_a_difficulty(monkeypatch, tmp_path):
    W, H = m.config.WIDTH, m.config.HEIGHT
    sx, sy, _ = m.menu_start_target(W, H)            # lies inside the "Mittel" circle
    tick = _fake_dwell_clock(monkeypatch)
    created, screens = [], []
    real_game, real_screen = m.Game, m.draw_difficulty_screen
    monkeypatch.setattr(m, "Game", lambda *a, **k: created.append(1) or real_game(*a, **k))
    monkeypatch.setattr(m, "draw_difficulty_screen",
                        lambda *a, **k: screens.append(1) or real_screen(*a, **k))
    run_loop(monkeypatch, tmp_path, FakeCamera([_pen_at_tip(sx, sy)]), 40, on_frame=tick,
             BG_MODE="off", SUIT_ENABLED=False, SABER_ENABLED=False, DWELL_SECONDS=1.0)
    assert screens                                   # Start was confirmed, the next screen came up
    assert created == []                             # holding still did not start a game


class _EndedGame:
    """A run that is over at once, so the game-over screen comes up without playing."""
    game_over = True
    score = 0

    def __init__(self, width, height, difficulty, sounds):
        self.difficulty = difficulty

    def update(self, blade, dt):
        pass

    def draw(self, frame):
        pass

    def draw_hud(self, frame):
        pass

    def draw_combo(self, frame):
        pass

    def reset(self):
        pass


def test_holding_still_after_startmenue_does_not_quit(monkeypatch, tmp_path):
    W, H = m.config.WIDTH, m.config.HEIGHT
    mx, my, _ = m.row_targets(W, H, 3)[1]                          # "Mittel"
    gx, gy, _ = m.row_targets(W, H, 3, y_frac=0.80, r=50)[1]       # "Startmenue"
    assert (gx, gy) == m.quit_target(W, H)[:2]                     # the same spot as the menu's Quit
    tick = _fake_dwell_clock(monkeypatch)
    monkeypatch.setattr(m, "initial_state", lambda identifier: m.STATE_DIFFICULTY)
    monkeypatch.setattr(m, "Game", _EndedGame)
    menus = []
    real_menu = m.draw_menu
    monkeypatch.setattr(m, "draw_menu", lambda *a, **k: menus.append(1) or real_menu(*a, **k))
    frames = [_pen_at_tip(mx, my)] * 14 + [_pen_at_tip(gx, gy)]
    ran = run_loop(monkeypatch, tmp_path, FakeCamera(frames), 60, on_frame=tick,
                   BG_MODE="off", SUIT_ENABLED=False, SABER_ENABLED=False, DWELL_SECONDS=1.0)
    assert menus                                     # the dwell on "Startmenue" opened the menu
    assert ran == 60                                 # the resting pen did not confirm "Quit"


# -- CR-17 (#69): a lost camera is opened again instead of ending the game ---
class _LostCamera(FakeCamera):
    """No camera is left (a switch and its rollback failed); reopen() works after a few tries."""

    def __init__(self, frames, back_after_tries=None):
        super().__init__(frames)
        self.is_open = False
        self.back_after = back_after_tries
        self.reopen_calls = 0

    def read(self):
        return super().read() if self.is_open else (False, None)

    def reopen(self):
        self.reopen_calls += 1
        if self.back_after is not None and self.reopen_calls >= self.back_after:
            self.is_open = True
        return self.is_open


def test_a_lost_camera_is_reopened_instead_of_ending_the_game(monkeypatch, tmp_path):
    camera = _LostCamera([green_frame()], back_after_tries=3)
    ran = run_loop(monkeypatch, tmp_path, camera, 12, BG_MODE="off", MAX_FRAME_FAILURES=3,
                   CAMERA_REOPEN_INTERVAL=0.0, CAMERA_LOST_TIMEOUT=60.0)
    assert ran == 12                                 # kept running until 'q'
    assert camera.reopen_calls == 3 and camera.good_reads > 0


def test_a_camera_that_never_comes_back_ends_the_game_after_the_timeout(monkeypatch, tmp_path):
    camera = _LostCamera([green_frame()])
    ran = run_loop(monkeypatch, tmp_path, camera, 50, BG_MODE="off",
                   CAMERA_REOPEN_INTERVAL=0.0, CAMERA_LOST_TIMEOUT=0.0)
    assert ran < 50 and camera.reopen_calls >= 1


# -- CR-18 (#70) / CR-13 (#65): name entry ------------------------------------
class _Stop(Exception):
    pass


class _NamingIdentifier:
    """Stands in for FaceIdentifier during the name entry: persons, scan and enroll."""

    def __init__(self, persons=(), enrolled=None):
        self.persons = list(persons)
        self.enroll_calls = []
        self._enrolled = enrolled

    def scan(self, frame):
        return None, 0.0, None, None

    def largest_face_box(self, frame, width=320):
        return None

    def enroll(self, faces_dir, name, crops):
        self.enroll_calls.append(name)
        return self._enrolled or face_id.Enrolled(len(crops), len(crops))


def _name_entry_loop(monkeypatch, tmp_path, identifier, text, frames=12):
    """Type ``text`` + Enter on the physical keyboard in the name entry; returns all drawn texts."""
    monkeypatch.setattr(m, "initial_state", lambda ident: m.STATE_NAME_ENTRY)
    texts = []
    real = m.draw_text
    monkeypatch.setattr(m, "draw_text",
                        lambda frame, t, *a, **k: texts.append(t) or real(frame, t, *a, **k))
    keys = {i + 1: ord(c) for i, c in enumerate(text)}
    keys[len(text) + 1] = 13                         # Enter

    def stop_after(n, mouse):                        # the letter q types here, so end by exception
        if n >= frames:
            raise _Stop
    try:
        run_loop(monkeypatch, tmp_path, FakeCamera([green_frame()]), frames, keys=keys,
                 on_frame=stop_after, identifier=identifier, BG_MODE="off",
                 SUIT_ENABLED=False, SABER_ENABLED=False, FACE_GREETING_SECONDS=0.0)
    except _Stop:
        pass
    return texts


def test_a_taken_name_is_refused_with_a_hint(monkeypatch, tmp_path):
    ident = _NamingIdentifier(persons=["Anna"])
    texts = _name_entry_loop(monkeypatch, tmp_path, ident, "anna")
    assert ident.enroll_calls == []
    assert any("vergeben" in t for t in texts)


def test_the_guest_name_is_refused_with_a_hint(monkeypatch, tmp_path):
    ident = _NamingIdentifier()
    texts = _name_entry_loop(monkeypatch, tmp_path, ident, "gast")
    assert ident.enroll_calls == []
    assert any("reserviert" in t for t in texts)


def test_a_free_name_is_enrolled(monkeypatch, tmp_path):
    ident = _NamingIdentifier(persons=["Anna"])
    _name_entry_loop(monkeypatch, tmp_path, ident, "ben")
    assert ident.enroll_calls == ["Ben"]


def test_a_failed_save_of_the_face_images_is_shown(monkeypatch, tmp_path):
    ident = _NamingIdentifier(enrolled=face_id.Enrolled(added=1, saved=0))
    texts = _name_entry_loop(monkeypatch, tmp_path, ident, "ben")
    assert any("gespeichert" in t for t in texts)


# -- CR-19 (#71): only the pen is protected from the suit, not other pen-coloured objects
def test_a_pen_coloured_object_elsewhere_is_not_protected_from_the_suit(monkeypatch, tmp_path):
    calls = _record_dress(monkeypatch)
    frame = with_pen(green_frame(), 75)
    cv2.rectangle(frame, (290, 140), (310, 160), (180, 0, 255), -1)    # a pink cushion on the player
    run_loop(monkeypatch, tmp_path, FakeCamera([frame]), 2,
             BG_MODE="greenscreen", SUIT_ENABLED=True)
    keep = calls[-1]
    assert keep[300, 75] > 0                                           # the pen is protected
    assert keep[150, 300] == 0                                         # the cushion is not
