"""Tests for the static lightsaber drawn at the cutting point. Headless."""

import numpy as np

import lightsaber

POINT = (300, 150)                                      # tracked pen point (cuts the fruit)


def test_no_saber_without_pen_point():
    assert lightsaber.saber_at(None) is None


def test_tip_is_the_cutting_point_and_the_saber_hangs_straight_down():
    s = lightsaber.saber_at(POINT, blade_length=120, hilt_length=50)
    assert s.tip == POINT
    assert s.hilt_top == (300, 150 + 120)                # blade straight down
    assert s.hilt_bottom == (300, 150 + 120 + 50)        # hilt below the blade


def test_default_lengths_scale_with_the_screen(monkeypatch):
    import config
    monkeypatch.setattr(config, "SCALE", 2.0)
    s = lightsaber.saber_at(POINT)
    assert s.hilt_top[1] - s.tip[1] == round(config.SABER_LENGTH * 2.0)


def _drawn(frame=None, color=(255, 0, 0)):
    frame = np.zeros((500, 600, 3), np.uint8) if frame is None else frame
    s = lightsaber.saber_at(POINT, blade_length=150, hilt_length=60)
    lightsaber.draw_saber(frame, s, color)
    return frame, s


def test_blade_is_a_thick_bright_beam():
    frame, s = _drawn()
    y = (s.tip[1] + s.hilt_top[1]) // 2
    assert frame[y, 300 - 5:300 + 6].min() > 200          # white-hot core at least 11 px wide


def test_blade_glows_in_the_saber_colour_around_the_core():
    frame, s = _drawn()
    y = (s.tip[1] + s.hilt_top[1]) // 2
    glow = frame[y, 300 + 16]                              # beside the core
    assert glow[0] > 80 and glow[0] > glow[2] + 40         # blue glow (BGR)
    assert (frame[480, 20] == 0).all()                     # far away untouched


def test_drawing_near_the_edge_or_without_pen_does_not_crash():
    frame = np.zeros((500, 600, 3), np.uint8)
    lightsaber.draw_saber(frame, lightsaber.saber_at((10, 480)), (0, 0, 255))
    lightsaber.draw_saber(frame, None, (0, 0, 255))


# --------------------------------------------------------------------------- #
# Hilt follows the hand
# --------------------------------------------------------------------------- #
import math

import cv2


def skin_with(*blobs, shape=(500, 600)):
    """Skin mask with filled circles: blobs = ((x, y), radius), ..."""
    skin = np.zeros(shape, np.uint8)
    for center, radius in blobs:
        cv2.circle(skin, center, radius, 255, -1)
    return skin


def _mid(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def test_no_hand_without_skin_or_too_small_or_too_far():
    assert lightsaber.find_hand(np.zeros((500, 600), np.uint8), POINT, radius=250) is None
    assert lightsaber.find_hand(skin_with(((330, 280), 4)), POINT, radius=250) is None
    assert lightsaber.find_hand(skin_with(((590, 480), 30)), POINT, radius=150) is None


def test_find_hand_takes_the_nearest_blob():
    skin = skin_with(((300, 260), 30), ((450, 420), 40))
    assert math.dist(lightsaber.find_hand(skin, POINT, radius=400), (300, 260)) < 3


def test_find_hand_ignores_the_face():
    skin = skin_with(((300, 90), 45), ((330, 300), 30))           # face above, hand below
    face = (255, 45, 90, 90)                                       # x, y, w, h
    hand = lightsaber.find_hand(skin, POINT, radius=250, exclude=face)
    assert math.dist(hand, (330, 300)) < 3


def test_hand_merged_with_a_big_skin_area_uses_the_part_next_to_the_pen():
    """Hand touching a bare arm or a skin-coloured object: not the centre of it all."""
    skin = skin_with(((330, 260), 30))
    cv2.rectangle(skin, (250, 280), (560, 480), 255, -1)           # large skin area attached below
    hand = lightsaber.find_hand(skin, POINT, radius=400)
    assert math.dist(hand, (330, 260)) < 45                        # near the hand, not ~(400, 380)


def _angle_deg(s):
    """Saber tilt in degrees: 0 = straight down, positive = hilt to the right."""
    dx, dy = s.hilt_top[0] - s.tip[0], s.hilt_top[1] - s.tip[1]
    return math.degrees(math.atan2(dx, dy))


def _tracker(**kw):
    kw.setdefault("blade_length", 120)
    kw.setdefault("hilt_length", 50)
    return lightsaber.SaberTracker(**kw)


def test_saber_stays_at_the_pen_and_changes_length_only_slowly():
    t = _tracker(smoothing=0.0)
    s = t.update(POINT, skin_with(((300, 420), 30)))               # hand far below
    assert s.tip == POINT                                          # stays at the pen
    assert abs(math.dist(s.tip, s.hilt_top) - 120) < 12            # barely changed in one frame
    assert abs(math.dist(s.hilt_top, s.hilt_bottom) - 50) < 2      # hilt length fixed


def test_hilt_settles_on_the_hand():
    t = _tracker(smoothing=0.0)
    hand = (300, 340)                                              # within SABER_HAND_RADIUS
    for _ in range(120):                                           # ~4 s at 30 fps
        s = t.update(POINT, skin_with((hand, 30)))
    assert math.dist(_mid(s.hilt_top, s.hilt_bottom), hand) < 6


def test_length_ignores_jitter_of_the_hand_detection():
    t = _tracker(smoothing=0.0)
    lengths = []
    for i in range(150):
        y = 360 + (25 if i % 2 else -25)                           # detection jumps +-25 px
        s = t.update(POINT, skin_with(((300, y), 30)))
        lengths.append(math.dist(s.tip, s.hilt_top))
    late = lengths[100:]
    assert max(late) - min(late) < 4                               # practically constant


def test_blade_length_has_limits():
    # adapt_rate 0.5 settles within a few frames (the default 5 % needs ~200)
    near = skin_with(((300, 175), 20))                             # hand right at the tip
    t = _tracker(smoothing=0.0, min_blade=80, max_blade=200, adapt_rate=0.5)
    for _ in range(20):
        s = t.update(POINT, near)
    assert 79 <= math.dist(s.tip, s.hilt_top) <= 81                # settled on the lower limit
    far = skin_with(((300, 495), 25))                              # hand very far away
    t = _tracker(smoothing=0.0, min_blade=80, max_blade=200, radius=500, adapt_rate=0.5)
    for _ in range(20):
        s = t.update(POINT, far)
    assert 199 <= math.dist(s.tip, s.hilt_top) <= 201              # settled on the upper limit


def test_skin_above_the_tip_is_not_taken_as_the_hand():
    s = _tracker(smoothing=0.0).update(POINT, skin_with(((300, 60), 35)))
    assert abs(_angle_deg(s)) < 1                                  # still hanging straight down


def test_direction_is_smoothed():
    t = _tracker(smoothing=0.5)
    s = t.update(POINT, skin_with(((300 + 81, 150 + 140), 30)))    # hand at about 30 deg
    assert 10 < _angle_deg(s) < 20                                 # only halfway at first


def test_a_single_wrong_frame_does_not_make_it_jump():
    t = _tracker(smoothing=0.0, confirm_frames=3)
    t.update(POINT, skin_with(((300, 300), 30)))                   # straight down
    s = t.update(POINT, skin_with(((420, 220), 30)))               # sudden ~60 deg, one frame
    assert abs(_angle_deg(s)) < 1


def test_a_lasting_change_is_taken_over():
    t = _tracker(smoothing=0.0, confirm_frames=3)
    t.update(POINT, skin_with(((300, 300), 30)))
    for _ in range(3):
        s = t.update(POINT, skin_with(((420, 220), 30)))
    assert _angle_deg(s) > 50


def test_without_a_hand_it_swings_back_slowly_not_at_once():
    t = _tracker(smoothing=0.0, keep_frames=0)
    t.update(POINT, skin_with(((380, 290), 30)))                   # ~30 deg
    s = t.update(POINT, np.zeros((500, 600), np.uint8))            # hand gone
    assert 5 < _angle_deg(s) < 30                                  # on its way back, no jump
    for _ in range(40):
        s = t.update(POINT, np.zeros((500, 600), np.uint8))
    assert abs(_angle_deg(s)) < 2                                  # vertical in the end


def test_no_pen_no_saber():
    assert _tracker().update(None, None) is None


def test_draw_tilted_saber():
    frame = np.full((500, 600, 3), 200, np.uint8)
    s = lightsaber.saber_toward(POINT, (0.5, math.sqrt(0.75)), blade_length=150, hilt_length=60)
    lightsaber.draw_saber(frame, s, (255, 0, 0))
    hx, hy = _mid(s.hilt_top, s.hilt_bottom)
    assert frame[int(hy), int(hx)].mean() < 120                    # dark hilt
    bx, by = _mid(s.tip, s.hilt_top)
    assert frame[int(by), int(bx)].min() > 200                     # bright blade


def test_skin_mask_finds_skin_but_not_clothes():
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:, :100] = (120, 150, 200)                               # BGR skin tone
    frame[:, 100:] = (200, 60, 40)                                 # blue jumper
    skin = lightsaber.skin_mask(frame)
    assert skin.shape == (100, 200)
    assert skin[50, 50] == 255
    assert skin[50, 150] == 0


# --------------------------------------------------------------------------- #
# Hilt on the glove the suit drew (#46): the pose socket instead of skin colour
# --------------------------------------------------------------------------- #
GRIP = (330, 300)                                   # palm centre of the glove holding the pen


def _glove(name, wrist_end, finger_end):
    """A glove part as SuitRenderer.last_parts records it (endpoints wrist -> fingers)."""
    return dict(name=name, asset="glove", endpoints=np.array((wrist_end, finger_end), np.float32))


def _hilt_centre(s):
    return _mid(s.hilt_top, s.hilt_bottom)


def test_glove_grips_are_the_palm_centres_of_the_drawn_gloves(monkeypatch):
    import config
    monkeypatch.setattr(config, "SABER_GLOVE_GRIP", 0.45)
    parts = [_glove("glove_l", (100, 200), (100, 300)),
             _glove("glove_r", (400, 250), (500, 250)),
             dict(name="forearm_l", asset="forearm",
                  endpoints=np.array(((0, 0), (9, 9)), np.float32))]
    grips = lightsaber.glove_grips(parts)
    assert set(grips) == {"l", "r"}                                # other parts are ignored
    assert math.dist(grips["l"], (100, 245)) < 1e-3                # 45 % from wrist to fingers
    assert math.dist(grips["r"], (445, 250)) < 1e-3


def test_no_glove_no_grip():
    assert lightsaber.glove_grips([]) == {}
    assert lightsaber.glove_grips(None) == {}


def test_hilt_sits_on_the_glove_from_the_first_frame():
    s = _tracker().update(POINT, None, hands={"r": GRIP})
    assert s.tip == POINT                                          # still cuts at the pen
    assert math.dist(_hilt_centre(s), GRIP) <= 1                   # no slow settling
    assert abs(math.dist(s.hilt_top, s.hilt_bottom) - 50) < 2      # hilt length unchanged


def test_hilt_follows_a_turning_hand_in_the_same_frame():
    t = _tracker(confirm_frames=4)
    t.update(POINT, None, hands={"r": (300, 300)})                 # straight down
    turned = (300 + 130, 150 + 75)                                 # ~60 deg, one frame later
    s = t.update(POINT, None, hands={"r": turned})
    assert math.dist(_hilt_centre(s), turned) <= 1                 # no 4-frame confirmation


def test_skin_is_not_computed_while_a_glove_holds_the_saber():
    calls = []

    def skin():
        calls.append(1)
        return skin_with(((300, 300), 30))

    t = _tracker()
    t.update(POINT, skin, hands={"r": GRIP})
    assert calls == []                                             # the expensive mask is skipped
    t.update(POINT, skin, hands={})                                # no glove: fall back to skin
    assert calls == [1]


def test_takes_the_glove_next_to_the_pen():
    other = (420, 280)                                             # the free hand, a bit further
    s = _tracker().update(POINT, None, hands={"l": other, "r": GRIP})
    assert math.dist(_hilt_centre(s), GRIP) <= 1


def test_a_slightly_closer_second_hand_does_not_take_the_saber(monkeypatch):
    import config
    monkeypatch.setattr(config, "SABER_HAND_SWITCH", 0.7)
    t = _tracker()
    t.update(POINT, None, hands={"r": GRIP})                       # pen hand ~153 px away
    near = (300 - 100, 150 + 100)                                  # other hand ~141 px: closer, not clearly
    s = t.update(POINT, None, hands={"r": GRIP, "l": near})
    assert math.dist(_hilt_centre(s), GRIP) <= 1


def test_a_clearly_closer_second_hand_takes_over(monkeypatch):
    import config
    monkeypatch.setattr(config, "SABER_HAND_SWITCH", 0.7)
    t = _tracker()
    t.update(POINT, None, hands={"r": GRIP})                       # ~153 px
    close = (300 - 60, 150 + 60)                                   # ~85 px < 0.7 * 153
    s = t.update(POINT, None, hands={"r": GRIP, "l": close})
    assert math.dist(_hilt_centre(s), close) <= 1


def test_a_glove_out_of_reach_is_ignored_and_skin_is_used():
    far = (590, 480)                                               # beyond the search radius
    t = _tracker(smoothing=0.0, radius=200)
    s = t.update(POINT, skin_with(((380, 290), 30)), hands={"r": far})
    expected = math.degrees(math.atan2(380 - 300, 290 - 150))      # towards the skin blob
    assert abs(_angle_deg(s) - expected) < 3


def test_losing_the_glove_does_not_make_the_saber_jump():
    t = _tracker(keep_frames=2)
    held = t.update(POINT, None, hands={"r": (300 + 81, 150 + 140)})    # ~30 deg, ~162 px
    s = t.update(POINT, np.zeros((500, 600), np.uint8), hands={})       # glove and skin gone
    assert abs(_angle_deg(s) - _angle_deg(held)) < 1                    # held for a moment
    assert abs(math.dist(s.tip, s.hilt_top) - math.dist(held.tip, held.hilt_top)) < 1


def test_glove_right_at_the_tip_still_centres_the_hilt_on_the_palm():
    grip = (305, 160)                                              # pen almost inside the fist
    s = _tracker().update(POINT, None, hands={"r": grip})          # closer than half a hilt (25)
    assert s.tip == POINT                                          # still cuts at the pen
    assert math.dist(_hilt_centre(s), grip) <= 1                   # hilt stays on the palm
    assert abs(math.dist(s.hilt_top, s.hilt_bottom) - 50) < 2


def test_hand_source_is_logged_once_per_lasting_change_not_per_frame(caplog):
    import logging
    t = _tracker(confirm_frames=3)
    no_skin = np.zeros((500, 600), np.uint8)
    with caplog.at_level(logging.INFO, logger="lightsaber"):
        for _ in range(6):
            t.update(POINT, None, hands={"r": GRIP})               # glove for 6 frames: one line
        for i in range(6):                                         # flicker never lasts 3 frames
            t.update(POINT, no_skin, hands={} if i % 2 else {"r": GRIP})
    lines = [r.getMessage() for r in caplog.records if r.name == "lightsaber"]
    assert len(lines) == 1 and "glove" in lines[0]
