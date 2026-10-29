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


def test_saber_runs_from_the_tip_to_the_hand():
    hand = (380, 290)
    s = lightsaber.saber_to_hand(POINT, hand, hilt_length=60)
    assert s.tip == POINT
    assert math.dist(_mid(s.hilt_top, s.hilt_bottom), hand) < 2   # hilt centred on the hand
    assert math.dist(POINT, s.hilt_top) < math.dist(POINT, s.hilt_bottom)   # blade side up


def test_saber_keeps_a_minimum_blade_when_the_hand_is_very_close():
    s = lightsaber.saber_to_hand(POINT, (300, 160), hilt_length=60, min_blade=50)
    assert math.dist(s.tip, s.hilt_top) >= 49


def test_tracker_follows_the_hand_smoothly():
    t = lightsaber.SaberTracker(smoothing=0.5, hilt_length=60)
    t.update(POINT, skin_with(((300, 300), 30)))
    s = t.update(POINT, skin_with(((340, 300), 30)))               # hand moved 40 px
    hx, _ = _mid(s.hilt_top, s.hilt_bottom)
    assert 310 < hx < 330                                          # halfway, no jump


def test_tracker_keeps_the_hand_through_a_short_dropout():
    t = lightsaber.SaberTracker(smoothing=0.0, hilt_length=60, keep_frames=3)
    t.update(POINT, skin_with(((380, 290), 30)))
    s = t.update(POINT, np.zeros((500, 600), np.uint8))            # hand missed for one frame
    assert math.dist(_mid(s.hilt_top, s.hilt_bottom), (380, 290)) < 2


def test_tracker_falls_back_to_hanging_straight_down_without_a_hand():
    t = lightsaber.SaberTracker(smoothing=0.0, hilt_length=60, keep_frames=2)
    t.update(POINT, skin_with(((380, 290), 30)))
    for _ in range(3):
        s = t.update(POINT, np.zeros((500, 600), np.uint8))
    assert s.hilt_top[0] == POINT[0] and s.hilt_top[1] > POINT[1]  # vertical again
    assert t.update(None, None) is None                            # no pen, no saber


def test_draw_tilted_saber_hilt_on_the_hand():
    frame = np.full((500, 600, 3), 200, np.uint8)
    s = lightsaber.saber_to_hand(POINT, (420, 330), hilt_length=70)
    lightsaber.draw_saber(frame, s, (255, 0, 0))
    hx, hy = _mid(s.hilt_top, s.hilt_bottom)
    assert frame[int(hy), int(hx)].mean() < 120                    # dark hilt at the hand
    bx, by = _mid(s.tip, s.hilt_top)
    assert frame[int(by), int(bx)].min() > 200                     # bright blade towards the tip


def test_hand_merged_with_a_big_skin_area_uses_the_part_next_to_the_pen():
    """Hand touching a bare arm or a skin-coloured object: not the centre of it all."""
    skin = skin_with(((330, 260), 30))
    cv2.rectangle(skin, (250, 280), (560, 480), 255, -1)           # large skin area attached below
    hand = lightsaber.find_hand(skin, POINT, radius=400)
    assert math.dist(hand, (330, 260)) < 45                        # near the hand, not ~(400, 380)
