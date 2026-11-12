"""Motion analysis for direction cuts (#33) -- synthetic position/time sequences, no camera."""

import math

import pytest

import config
from motion import (CUT_NONE, CUT_OK, CUT_TOO_SLOW, CUT_WRONG, DIRECTIONS, DOWN, LEFT, RIGHT,
                    UP, StrokeTracker, angle_deviation, classify_stroke)

FIELD = 1000.0                    # field width in px for the examples
CENTER = (500.0, 500.0)


def _path(direction, speed, fps, seconds=0.5, deviation_deg=0.0, start=CENTER):
    """Samples of a constant-velocity pen path: ``speed`` in field widths per second,
    rotated by ``deviation_deg`` away from ``direction``. Returns [(point, t)]."""
    a = math.atan2(direction.vy, direction.vx) + math.radians(deviation_deg)
    vx, vy = math.cos(a) * speed * FIELD, math.sin(a) * speed * FIELD
    n = int(round(seconds * fps)) + 1
    return [((start[0] + vx * k / fps, start[1] + vy * k / fps), k / fps) for k in range(n)]


def _tracker(samples):
    tr = StrokeTracker(FIELD)
    for p, t in samples:
        tr.add(p, t)
    return tr


def _verdict(samples, direction, **kw):
    return classify_stroke(_tracker(samples).stroke(), direction, **kw)


# -- directions ---------------------------------------------------------------
def test_four_distinct_directions():
    assert len(DIRECTIONS) == 4
    assert {d.name for d in DIRECTIONS} == {"right", "left", "down", "up"}
    assert (RIGHT.vx, RIGHT.vy) == (1, 0) and (LEFT.vx, LEFT.vy) == (-1, 0)
    assert (DOWN.vx, DOWN.vy) == (0, 1) and (UP.vx, UP.vy) == (0, -1)   # image y grows downward


def test_direction_labels_are_human_readable():
    assert all(d.label for d in DIRECTIONS)
    assert len({d.label for d in DIRECTIONS}) == 4


# -- speed unit and direction from timestamps -------------------------------
def test_speed_is_in_field_widths_per_second():
    stroke = _tracker(_path(RIGHT, 2.0, 30)).stroke()
    assert stroke.speed == pytest.approx(2.0, rel=1e-6)
    assert stroke.angle == pytest.approx(0.0, abs=1e-9)


def test_speed_scales_with_field_width_not_pixels():
    samples = _path(RIGHT, 2.0, 30)
    small = StrokeTracker(FIELD / 2)
    for p, t in samples:
        small.add(p, t)
    assert small.stroke().speed == pytest.approx(4.0, rel=1e-6)   # same px/s on a smaller field


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_stroke_angle_matches_the_direction_of_motion(direction):
    stroke = _tracker(_path(direction, 1.5, 30)).stroke()
    assert angle_deviation(stroke, direction) == pytest.approx(0.0, abs=1e-6)


def test_stroke_needs_two_samples():
    tr = StrokeTracker(FIELD)
    assert tr.stroke() is None and tr.segment() is None
    tr.add((100, 100), 0.0)
    assert tr.stroke() is None and tr.segment() is None
    tr.add((140, 100), 0.03)
    assert tr.stroke() is not None and tr.segment() == ((100, 100), (140, 100))


def test_segment_is_between_the_last_two_valid_positions():
    tr = _tracker([((0, 0), 0.0), ((10, 0), 0.03), ((30, 5), 0.06)])
    assert tr.segment() == ((10, 0), (30, 5))


# -- angle deviation ------------------------------------------------------------
@pytest.mark.parametrize("direction,opposite", [(RIGHT, LEFT), (DOWN, UP)])
def test_opposite_direction_is_180_degrees_off(direction, opposite):
    stroke = _tracker(_path(opposite, 1.5, 30)).stroke()
    assert angle_deviation(stroke, direction) == pytest.approx(180.0, abs=1e-6)


@pytest.mark.parametrize("deviation", [10, 29, 45, 90, -10, -29, -45, -90, 170])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_deviation_is_the_unsigned_angle(direction, deviation):
    stroke = _tracker(_path(direction, 1.5, 30, deviation_deg=deviation)).stroke()
    assert angle_deviation(stroke, direction) == pytest.approx(abs(deviation), abs=1e-6)


def test_deviation_wraps_around_correctly():
    # a stroke at 350 degrees and a target at 10 degrees are 20 degrees apart, not 340
    a = math.radians(350)
    samples = [((500 + math.cos(a) * 30 * k, 500 + math.sin(a) * 30 * k), k * 0.03)
               for k in range(5)]
    stroke = _tracker(samples).stroke()
    target = type(RIGHT)("t", "t", math.cos(math.radians(10)), math.sin(math.radians(10)))
    assert angle_deviation(stroke, target) == pytest.approx(20.0, abs=1e-6)


# -- the cut decision -----------------------------------------------------------
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_fast_stroke_in_the_arrow_direction_is_a_valid_cut(direction):
    assert _verdict(_path(direction, 2.0, 30), direction).kind == CUT_OK


@pytest.mark.parametrize("direction,opposite", [(RIGHT, LEFT), (LEFT, RIGHT), (DOWN, UP), (UP, DOWN)])
def test_opposite_direction_on_the_same_axis_is_wrong(direction, opposite):
    assert _verdict(_path(opposite, 2.0, 30), direction).kind == CUT_WRONG


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_perpendicular_and_diagonal_strokes_are_wrong(direction):
    assert _verdict(_path(direction, 2.0, 30, deviation_deg=90), direction).kind == CUT_WRONG
    assert _verdict(_path(direction, 2.0, 30, deviation_deg=45), direction).kind == CUT_WRONG


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_tolerance_boundary(direction):
    tol = config.CUT_TOLERANCE_DEG
    assert _verdict(_path(direction, 2.0, 30, deviation_deg=tol - 1), direction).kind == CUT_OK
    assert _verdict(_path(direction, 2.0, 30, deviation_deg=-(tol - 1)), direction).kind == CUT_OK
    assert _verdict(_path(direction, 2.0, 30, deviation_deg=tol + 1), direction).kind == CUT_WRONG
    assert _verdict(_path(direction, 2.0, 30, deviation_deg=-(tol + 1)), direction).kind == CUT_WRONG


def test_tolerance_is_adjustable():
    samples = _path(RIGHT, 2.0, 30, deviation_deg=40)
    assert _verdict(samples, RIGHT).kind == CUT_WRONG
    assert _verdict(samples, RIGHT, tolerance_deg=45).kind == CUT_OK


def test_slow_stroke_is_too_slow_even_in_the_right_direction():
    slow = config.CUT_MIN_SPEED * 0.5
    verdict = _verdict(_path(RIGHT, slow, 30), RIGHT)
    assert verdict.kind == CUT_TOO_SLOW and verdict.speed == pytest.approx(slow, rel=1e-6)


def test_speed_threshold_boundary():
    v = config.CUT_MIN_SPEED
    assert _verdict(_path(RIGHT, v * 1.05, 30), RIGHT).kind == CUT_OK
    assert _verdict(_path(RIGHT, v * 0.95, 30), RIGHT).kind == CUT_TOO_SLOW


def test_a_slow_wrong_direction_stroke_is_just_too_slow():
    # no penalty feedback for slow touches: speed is checked before the direction
    assert _verdict(_path(LEFT, config.CUT_MIN_SPEED * 0.3, 30), RIGHT).kind == CUT_TOO_SLOW


def test_standing_still_is_no_cut():
    tr = _tracker([((300, 300), k * 0.033) for k in range(12)])
    verdict = classify_stroke(tr.stroke(), RIGHT)
    assert verdict.kind == CUT_TOO_SLOW and verdict.speed == 0.0


def test_no_stroke_means_no_cut():
    assert classify_stroke(None, RIGHT).kind == CUT_NONE


def test_min_speed_is_adjustable():
    samples = _path(RIGHT, 0.5, 30)
    assert _verdict(samples, RIGHT).kind == CUT_TOO_SLOW
    assert _verdict(samples, RIGHT, min_speed=0.3).kind == CUT_OK


# -- tracking gaps --------------------------------------------------------------
def test_lost_pen_starts_a_new_track():
    tr = StrokeTracker(FIELD)
    for p, t in _path(RIGHT, 2.0, 30, seconds=0.2):
        tr.add(p, t)
    assert tr.stroke() is not None
    tr.add(None, 0.25)                                  # pen not found in this frame
    assert tr.stroke() is None and tr.segment() is None
    tr.add((900, 500), 0.28)                            # reappears far away
    assert tr.stroke() is None and tr.segment() is None   # one point: nothing to cut with
    tr.add((930, 500), 0.31)
    seg = tr.segment()
    assert seg == ((900, 500), (930, 500))              # only the new, short segment
    assert tr.stroke().speed < 2.0                      # no streak across the gap


def test_reappearing_pen_never_produces_a_cross_field_stroke():
    tr = StrokeTracker(FIELD)
    tr.add((50, 500), 0.0)
    tr.add((80, 500), 0.03)
    tr.add(None, 0.06)
    tr.add((950, 500), 0.09)
    tr.add((950, 500), 0.12)
    assert classify_stroke(tr.stroke(), RIGHT).kind == CUT_TOO_SLOW


def test_long_time_gap_between_samples_restarts_the_track():
    tr = StrokeTracker(FIELD)
    tr.add((100, 500), 0.0)
    tr.add((130, 500), 0.03)
    tr.add((900, 500), 0.03 + config.CUT_MAX_GAP + 0.5)   # frames were dropped for a while
    assert tr.stroke() is None and tr.segment() is None


def test_short_gap_below_the_limit_keeps_the_track():
    tr = StrokeTracker(FIELD)
    tr.add((100, 500), 0.0)
    tr.add((160, 500), config.CUT_MAX_GAP * 0.5)
    assert tr.segment() is not None


# -- invalid timestamps ---------------------------------------------------------
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), "x"])
def test_unusable_timestamps_are_ignored(bad):
    tr = StrokeTracker(FIELD)
    assert tr.add((100, 100), 0.0) is True
    assert tr.add((200, 100), bad) is False
    assert tr.rejected == 1 and tr.segment() is None    # the sample was dropped, track intact
    assert tr.add((140, 100), 0.03) is True
    assert tr.segment() == ((100, 100), (140, 100))


def test_repeated_timestamp_is_ignored():
    tr = StrokeTracker(FIELD)
    tr.add((100, 100), 1.0)
    assert tr.add((150, 100), 1.0) is False             # same time, other position
    assert tr.rejected == 1 and tr.stroke() is None


def test_timestamp_going_backwards_is_ignored():
    tr = StrokeTracker(FIELD)
    tr.add((100, 100), 1.0)
    tr.add((140, 100), 1.03)
    assert tr.add((300, 100), 1.01) is False
    assert tr.segment() == ((100, 100), (140, 100))
    assert tr.rejected == 1


def test_invalid_timestamps_never_raise_or_divide_by_zero():
    tr = StrokeTracker(FIELD)
    for p, t in [((1, 1), 5.0), ((2, 2), 5.0), ((3, 3), 4.0), ((4, 4), float("nan"))]:
        tr.add(p, t)
    assert tr.stroke() is None


def test_reset_clears_everything():
    tr = _tracker(_path(RIGHT, 2.0, 30))
    tr.reset()
    assert tr.stroke() is None and tr.segment() is None


# -- sampling-rate independence ---------------------------------------------------
@pytest.mark.parametrize("fps", [8, 10, 15, 24, 30, 60, 120])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_same_motion_gives_the_same_decisions_at_any_frame_rate(direction, fps):
    assert _verdict(_path(direction, 2.0, fps), direction).kind == CUT_OK
    assert _verdict(_path(direction, 2.0, fps), {RIGHT: LEFT, LEFT: RIGHT, DOWN: UP, UP: DOWN}[direction]).kind == CUT_WRONG
    assert _verdict(_path(direction, 0.4, fps), direction).kind == CUT_TOO_SLOW
    assert _verdict(_path(direction, 2.0, fps, deviation_deg=20), direction).kind == CUT_OK
    assert _verdict(_path(direction, 2.0, fps, deviation_deg=40), direction).kind == CUT_WRONG


@pytest.mark.parametrize("fps", [8, 10, 15, 30, 60, 120])
def test_measured_speed_does_not_depend_on_the_frame_rate(fps):
    assert _tracker(_path(RIGHT, 2.0, fps)).stroke().speed == pytest.approx(2.0, rel=1e-6)


def test_a_fast_cut_with_only_two_samples_is_judged():
    tr = StrokeTracker(FIELD)
    tr.add((200, 500), 0.0)
    tr.add((500, 500), 0.05)                            # 300 px in 50 ms = 6 field widths/s
    verdict = classify_stroke(tr.stroke(), RIGHT)
    assert verdict.kind == CUT_OK and verdict.speed == pytest.approx(6.0)


def test_only_the_recent_time_window_counts_for_the_direction():
    # a long swipe right that turns sharply down: the decision follows the recent motion
    samples = []
    t = 0.0
    for k in range(10):
        samples.append(((100 + 60 * k, 300), t)); t += 1 / 30
    for k in range(1, 10):
        samples.append(((100 + 60 * 9, 300 + 60 * k), t)); t += 1 / 30
    verdict = _verdict(samples, DOWN)
    assert verdict.kind == CUT_OK
    assert _verdict(samples, RIGHT).kind == CUT_WRONG


def test_samples_older_than_the_window_do_not_dilute_the_speed():
    # standing still for a second, then a sudden fast move: judged by the recent move
    tr = StrokeTracker(FIELD)
    for k in range(30):
        tr.add((300, 500), k / 30)
    t = 1.0
    for k in range(1, 5):
        tr.add((300 + 80 * k, 500), t + k / 30)
    assert classify_stroke(tr.stroke(), RIGHT).kind == CUT_OK


# -- impossible jumps (the tracker latching onto another object) ----------------------
def test_an_impossible_jump_starts_a_new_track():
    tr = StrokeTracker(FIELD)
    tr.add((100, 500), 0.0)
    tr.add((130, 500), 0.03)
    tr.add((900, 500), 0.06)                            # 770 px in 30 ms: ~26 field widths/s
    assert tr.segment() is None and tr.stroke() is None
    tr.add((860, 500), 0.09)                            # the real motion continues from here
    stroke = tr.stroke()
    assert stroke.dx == -40 and classify_stroke(stroke, LEFT).kind in (CUT_OK, CUT_TOO_SLOW)


def test_the_jump_does_not_leak_into_the_direction_of_the_next_stroke():
    tr = StrokeTracker(FIELD)
    tr.add((40, 40), 0.0)
    tr.add((40, 40), 0.01)
    for k in range(8):                                  # a fast swipe left after the jump
        tr.add((800 - 64 * k, 377), 0.02 + 0.01 * k)
    assert classify_stroke(tr.stroke(), LEFT).kind == CUT_OK


def test_fast_but_plausible_motion_keeps_the_track():
    tr = StrokeTracker(FIELD)
    tr.add((100, 500), 0.0)
    tr.add((340, 500), 0.03)                            # 240 px in 30 ms: 8 field widths/s
    assert tr.segment() is not None


def test_the_plausibility_limit_is_adjustable():
    tr = StrokeTracker(FIELD, max_speed=100.0)
    tr.add((100, 500), 0.0)
    tr.add((900, 500), 0.03)
    assert tr.segment() is not None
