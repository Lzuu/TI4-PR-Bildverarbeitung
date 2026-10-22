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
# Saber geometry over time
# --------------------------------------------------------------------------- #
def test_saber_hilt_on_the_pen_and_blade_beyond_its_upper_end():
    saber = lightsaber.SaberTracker(blade_length=200)
    s = saber.update(pen_mask())
    (gx, gy), (ex, ey), (tx, ty) = s.grip, s.emitter, s.tip
    assert gy > ey > ty                                 # grip below, emitter, tip on top
    assert abs(ey - (250 - 60)) < 6                     # emitter = upper end of the pen
    assert abs((ey - ty) - 200) < 2                     # blade length


def test_saber_does_not_flip_when_the_pen_turns_past_horizontal():
    saber = lightsaber.SaberTracker(blade_length=200, smoothing=0.0)
    saber.update(pen_mask(angle_deg=80))                # almost horizontal, blade to the right
    s = saber.update(pen_mask(angle_deg=100))           # just past horizontal
    assert s.tip[0] > s.emitter[0]                      # still pointing right, no jump to the left


def test_round_blob_keeps_the_last_direction():
    saber = lightsaber.SaberTracker(blade_length=200, smoothing=0.0)
    saber.update(pen_mask(angle_deg=40))
    mask = np.zeros((500, 600), np.uint8)
    cv2.circle(mask, (300, 250), 30, 255, -1)           # pen seen end-on
    s = saber.update(mask)
    assert abs(abs(_angle_of(lightsaber.unit(s.tip, s.emitter))) - 40) < 3


def test_direction_is_smoothed_against_jitter():
    saber = lightsaber.SaberTracker(blade_length=200, smoothing=0.5)
    saber.update(pen_mask(angle_deg=0))
    s = saber.update(pen_mask(angle_deg=20))
    angle = abs(_angle_of(lightsaber.unit(s.tip, s.emitter)))
    assert 5 < angle < 15                               # halfway, not the full jump


def test_no_saber_without_pen_and_fresh_start_afterwards():
    saber = lightsaber.SaberTracker(blade_length=200, smoothing=0.5)
    saber.update(pen_mask(angle_deg=60))
    assert saber.update(np.zeros((500, 600), np.uint8)) is None
    s = saber.update(pen_mask(angle_deg=0))             # no slow turn from the old angle
    assert abs(_angle_of(lightsaber.unit(s.tip, s.emitter))) < 3


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def test_draw_blade_glows_in_the_saber_colour():
    frame = np.zeros((500, 600, 3), np.uint8)
    s = lightsaber.SaberTracker(blade_length=200).update(pen_mask())
    lightsaber.draw_saber(frame, s, (255, 0, 0))        # blue saber (BGR)
    mid = ((s.emitter[0] + s.tip[0]) // 2, (s.emitter[1] + s.tip[1]) // 2)
    assert frame[mid[1], mid[0]].min() > 200            # bright white core
    glow = frame[mid[1], mid[0] + 12]                   # beside the core
    assert glow[0] > 60 and glow[0] > glow[2] + 30      # blue glow
    assert (frame[480, 20] == 0).all()                  # far away untouched


def test_draw_hilt_over_the_pen():
    frame = np.full((500, 600, 3), 200, np.uint8)
    s = lightsaber.SaberTracker(blade_length=200).update(pen_mask())
    lightsaber.draw_saber(frame, s, (255, 0, 0))
    gx, gy = s.grip
    ex, ey = s.emitter
    hx, hy = (gx + ex) // 2, (gy + ey) // 2
    assert frame[hy, hx].mean() < 150                   # dark metal hilt where the pen is


def test_draw_partly_outside_the_frame_does_not_crash():
    frame = np.zeros((500, 600, 3), np.uint8)
    s = lightsaber.SaberTracker(blade_length=400).update(pen_mask(center=(30, 80)))
    lightsaber.draw_saber(frame, s, (0, 0, 255))
    lightsaber.draw_saber(frame, None, (0, 0, 255))     # no pen: nothing happens
