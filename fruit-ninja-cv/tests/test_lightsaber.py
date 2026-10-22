"""Tests for drawing the tracked pen as a lightsaber. Headless, synthetic masks."""

import math

import cv2
import numpy as np

import lightsaber


def pen_mask(center=(300, 250), length=120, width=20, angle_deg=0, shape=(500, 600)):
    """Mask of an elongated pen; angle 0 = vertical, positive = tilted clockwise."""
    mask = np.zeros(shape, np.uint8)
    box = cv2.boxPoints(((center[0], center[1]), (width, length), angle_deg))
    cv2.fillPoly(mask, [box.astype(np.int32)], 255)
    return mask


def _angle_of(direction):
    """Angle of a direction vector against 'straight up', in degrees (clockwise +)."""
    dx, dy = direction
    return math.degrees(math.atan2(dx, -dy))


# --------------------------------------------------------------------------- #
# Pen axis from the colour mask
# --------------------------------------------------------------------------- #
def test_no_axis_without_pen():
    assert lightsaber.pen_axis(np.zeros((500, 600), np.uint8)) is None


def test_tiny_blob_is_not_a_pen():
    mask = np.zeros((500, 600), np.uint8)
    mask[100:104, 100:104] = 255
    assert lightsaber.pen_axis(mask) is None


def test_vertical_pen_points_up():
    axis = lightsaber.pen_axis(pen_mask())
    assert abs(axis.center[0] - 300) < 2 and abs(axis.center[1] - 250) < 2
    assert abs(axis.length - 120) < 6
    assert abs(_angle_of(axis.direction)) < 3          # straight up, not down


def test_tilted_pen_direction():
    axis = lightsaber.pen_axis(pen_mask(angle_deg=30))
    assert abs(abs(_angle_of(axis.direction)) - 30) < 3


def test_direction_is_a_unit_vector_pointing_upwards():
    for angle in (-60, -20, 0, 45, 70):
        dx, dy = lightsaber.pen_axis(pen_mask(angle_deg=angle)).direction
        assert abs(math.hypot(dx, dy) - 1) < 1e-6
        assert dy < 0                                   # the blade grows upwards


def test_round_blob_has_no_reliable_direction():
    mask = np.zeros((500, 600), np.uint8)
    cv2.circle(mask, (300, 250), 30, 255, -1)
    axis = lightsaber.pen_axis(mask)
    assert axis is not None and axis.direction is None


# --------------------------------------------------------------------------- #
# Saber geometry: tip = tracked pen point, blade down to the hand, hilt on the hand
# --------------------------------------------------------------------------- #
PEN = (300, 250)                                        # tracked pen point (= pen centre)


def skin_blob(center, radius=30, shape=(500, 600)):
    skin = np.zeros(shape, np.uint8)
    cv2.circle(skin, center, radius, 255, -1)
    return skin


def _mid(a, b):
    return ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2)


def test_blade_tip_is_the_tracked_pen_point():
    s = lightsaber.SaberTracker().update(pen_mask(), PEN)
    assert s.tip == PEN                                 # the point that cuts the fruit


def test_blade_goes_down_and_the_hilt_sits_on_the_hand():
    hand = (300, 360)                                   # skin just below the pen
    s = lightsaber.SaberTracker(smoothing=0.0).update(pen_mask(), PEN, skin_blob(hand))
    hilt_center = _mid(s.hilt_top, s.hilt_bottom)
    assert math.dist(hilt_center, hand) < 12            # hilt on the hand
    assert s.tip[1] < s.hilt_top[1] < s.hilt_bottom[1]  # tip on top, blade down, pommel below


def test_without_a_hand_the_hilt_is_at_the_lower_end_of_the_pen():
    s = lightsaber.SaberTracker(smoothing=0.0).update(pen_mask(), PEN)
    hilt_center = _mid(s.hilt_top, s.hilt_bottom)
    assert math.dist(hilt_center, (300, 250 + 60)) < 8   # pen centre + half its length


def test_skin_far_away_like_the_face_is_not_the_hand():
    face = skin_blob((300, 60), radius=50)              # above the pen
    s = lightsaber.SaberTracker(smoothing=0.0).update(pen_mask(), PEN, face)
    hilt_center = _mid(s.hilt_top, s.hilt_bottom)
    assert math.dist(hilt_center, (300, 310)) < 8


def test_tilted_pen_hilt_on_the_hand_below_it():
    hand = (300 - 50, 250 + 87)                         # lower end of a pen tilted by 30 deg
    s = lightsaber.SaberTracker(smoothing=0.0).update(pen_mask(angle_deg=30), PEN,
                                                      skin_blob(hand, radius=25))
    assert math.dist(_mid(s.hilt_top, s.hilt_bottom), hand) < 15


def test_saber_does_not_flip_when_the_pen_turns_past_horizontal():
    saber = lightsaber.SaberTracker(smoothing=0.0)
    saber.update(pen_mask(angle_deg=80), PEN)           # almost horizontal, tip to the right
    s = saber.update(pen_mask(angle_deg=100), PEN)      # just past horizontal
    assert s.hilt_bottom[0] < s.tip[0]                  # hilt still on the left, no jump


def test_round_blob_keeps_the_last_direction():
    saber = lightsaber.SaberTracker(smoothing=0.0)
    saber.update(pen_mask(angle_deg=40), PEN)
    mask = np.zeros((500, 600), np.uint8)
    cv2.circle(mask, PEN, 30, 255, -1)                  # pen seen end-on
    s = saber.update(mask, PEN)
    assert abs(abs(_angle_of(lightsaber.unit(s.tip, s.hilt_top))) - 40) < 3


def test_direction_is_smoothed_against_jitter():
    saber = lightsaber.SaberTracker(smoothing=0.5)
    saber.update(pen_mask(angle_deg=0), PEN)
    s = saber.update(pen_mask(angle_deg=20), PEN)
    angle = abs(_angle_of(lightsaber.unit(s.tip, s.hilt_top)))
    assert 5 < angle < 15                               # halfway, not the full jump


def test_no_saber_without_pen_and_fresh_start_afterwards():
    saber = lightsaber.SaberTracker(smoothing=0.5)
    saber.update(pen_mask(angle_deg=60), PEN)
    assert saber.update(np.zeros((500, 600), np.uint8), None) is None
    s = saber.update(pen_mask(angle_deg=0), PEN)        # no slow turn from the old angle
    assert abs(_angle_of(lightsaber.unit(s.tip, s.hilt_top))) < 3


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def test_draw_blade_glows_in_the_saber_colour():
    frame = np.zeros((500, 600, 3), np.uint8)
    s = lightsaber.SaberTracker().update(pen_mask(), PEN, skin_blob((300, 420)))
    lightsaber.draw_saber(frame, s, (255, 0, 0))        # blue saber (BGR)
    mid = _mid(s.tip, s.hilt_top)
    assert frame[mid[1], mid[0]].min() > 200            # bright white core
    glow = frame[mid[1], mid[0] + 12]                   # beside the core
    assert glow[0] > 60 and glow[0] > glow[2] + 30      # blue glow
    assert (frame[480, 20] == 0).all()                  # far away untouched


def test_draw_hilt_on_the_hand():
    frame = np.full((500, 600, 3), 200, np.uint8)
    s = lightsaber.SaberTracker().update(pen_mask(), PEN, skin_blob((300, 400)))
    lightsaber.draw_saber(frame, s, (255, 0, 0))
    hx, hy = _mid(s.hilt_top, s.hilt_bottom)
    assert frame[hy, hx].mean() < 150                   # dark metal hilt


def test_draw_partly_outside_the_frame_does_not_crash():
    frame = np.zeros((500, 600, 3), np.uint8)
    s = lightsaber.SaberTracker().update(pen_mask(center=(30, 450)), (30, 450))
    lightsaber.draw_saber(frame, s, (0, 0, 255))
    lightsaber.draw_saber(frame, None, (0, 0, 255))     # no pen: nothing happens
