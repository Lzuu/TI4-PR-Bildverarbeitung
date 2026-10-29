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


def test_hilt_is_dark_metal_below_the_blade():
    frame, s = _drawn(np.full((500, 600, 3), 200, np.uint8))
    y = (s.hilt_top[1] + s.hilt_bottom[1]) // 2
    assert frame[y, 300].mean() < 120


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


def test_find_hand_is_the_skin_blob_near_the_pen_tip():
    hand = lightsaber.find_hand(skin_with(((330, 280), 35)), POINT, radius=250)
    assert math.dist(hand, (330, 280)) < 3


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


def test_saber_always_has_the_same_length():
    near = _tracker(smoothing=0.0).update(POINT, skin_with(((310, 210), 30)))
    far = _tracker(smoothing=0.0).update(POINT, skin_with(((420, 360), 30)))
    for s in (near, far):
        assert s.tip == POINT                                      # stays at the pen
        assert abs(math.dist(s.tip, s.hilt_top) - 120) < 2         # fixed blade
        assert abs(math.dist(s.hilt_top, s.hilt_bottom) - 50) < 2  # fixed hilt


def test_saber_points_towards_the_hand():
    s = _tracker(smoothing=0.0).update(POINT, skin_with(((380, 290), 30)))
    expected = math.degrees(math.atan2(380 - 300, 290 - 150))
    assert abs(_angle_deg(s) - expected) < 3


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
