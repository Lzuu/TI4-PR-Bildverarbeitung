"""Pen motion analysis for the direction cuts of the challenge mode (#33).

Pure functions and classes without any camera or GUI: they only see pen positions
and capture timestamps, so every rule can be tested with synthetic sequences.

* ``StrokeTracker`` keeps the recent, *valid* samples of the pen track and derives the
  current stroke from them: direction, and speed in **field widths per second**, i.e.
  independent of the render resolution.
* ``classify_stroke`` is the decision for one fruit: valid cut, too slow, or wrong
  direction.

Rules that keep the decision honest:

* Everything is based on **time**, not on frame counts. The direction and the speed come
  from the net movement within a short time window (``CUT_WINDOW``, at least the last two
  samples), so the same motion gives the same decision at 8 or 120 frames per second.
* A lost pen (``None``) ends the track. A reacquired pen starts a new one, so it can never
  produce a cut across the whole field (cf. #41, CR-01). A long time gap between two
  samples (``CUT_MAX_GAP``) is treated the same way.
* A movement faster than any hand can make (``CUT_MAX_SPEED``, e.g. the tracker latching
  onto another pink object for one frame) is treated like a lost pen: it starts a new
  track and can neither cut a fruit nor tip the direction of the next stroke.
* Unusable timestamps (missing, not finite, repeated or going backwards) are dropped and
  counted in ``rejected``; they never raise and never divide by zero.
* Coordinates are those of the picture the player sees (the camera image is mirrored
  *before* tracking), so an arrow and the pen movement refer to the same screen.
"""

import math
from collections import deque, namedtuple

import config

# Directions as seen on the screen. Image y grows downward, so "down" is +y.
Direction = namedtuple("Direction", "name label vx vy")
RIGHT = Direction("right", "links nach rechts", 1, 0)
LEFT = Direction("left", "rechts nach links", -1, 0)
DOWN = Direction("down", "oben nach unten", 0, 1)
UP = Direction("up", "unten nach oben", 0, -1)
DIRECTIONS = (RIGHT, LEFT, DOWN, UP)

CUT_OK = "ok"
CUT_TOO_SLOW = "too_slow"
CUT_WRONG = "wrong_direction"
CUT_NONE = "none"                 # no stroke at all (pen just appeared / not seen)

# dx, dy in px, dt in seconds, speed in field widths per second, angle in radians
Stroke = namedtuple("Stroke", "dx dy dt speed angle")
Verdict = namedtuple("Verdict", "kind speed deviation")


class StrokeTracker:
    MAX_SAMPLES = 64

    def __init__(self, field_width, window=None, max_gap=None, max_speed=None):
        self.field_width = float(field_width)
        self.window = config.CUT_WINDOW if window is None else window
        self.max_gap = config.CUT_MAX_GAP if max_gap is None else max_gap
        self.max_speed = config.CUT_MAX_SPEED if max_speed is None else max_speed
        self.rejected = 0                  # samples dropped for an unusable timestamp
        self._samples = deque(maxlen=self.MAX_SAMPLES)   # (t, (x, y)), t strictly increasing

    def reset(self):
        self._samples.clear()

    def add(self, point, t):
        """Add the pen position seen at time ``t`` (seconds), or ``None`` if lost.

        Returns True if the sample was used. A lost pen ends the track (returns False);
        an unusable timestamp is dropped and counted in ``rejected``.
        """
        if point is None:
            self._samples.clear()
            return False
        try:
            t = float(t)
            valid = math.isfinite(t)
        except (TypeError, ValueError):
            valid = False
        if valid and self._samples and t <= self._samples[-1][0]:
            valid = False                  # repeated or going backwards
        if not valid:
            self.rejected += 1
            return False
        if self._samples:
            t_prev, (x_prev, y_prev) = self._samples[-1]
            gap = t - t_prev
            jump = math.hypot(point[0] - x_prev, point[1] - y_prev) / gap / self.field_width
            if gap > self.max_gap or jump > self.max_speed:
                self._samples.clear()      # frames lost for too long / impossible jump
        self._samples.append((t, tuple(point)))
        return True

    def segment(self):
        """``(prev, cur)``: the last two valid positions of the current track, or None."""
        if len(self._samples) < 2:
            return None
        return self._samples[-2][1], self._samples[-1][1]

    def stroke(self):
        """The current stroke, or None while fewer than two samples are known."""
        n = len(self._samples)
        if n < 2:
            return None
        t_end, (x_end, y_end) = self._samples[-1]
        cutoff = t_end - self.window
        start = n - 2                      # always at least the previous sample
        while start > 0 and self._samples[start - 1][0] >= cutoff:
            start -= 1
        t0, (x0, y0) = self._samples[start]
        dx, dy, dt = x_end - x0, y_end - y0, t_end - t0
        length = math.hypot(dx, dy)
        speed = length / dt / self.field_width
        angle = math.atan2(dy, dx) if length > 0 else 0.0
        return Stroke(dx, dy, dt, speed, angle)


def angle_deviation(stroke, direction):
    """Unsigned angle in degrees (0..180) between the stroke and ``direction``."""
    if stroke.dx == 0 and stroke.dy == 0:
        return 0.0
    cross = stroke.dx * direction.vy - stroke.dy * direction.vx
    dot = stroke.dx * direction.vx + stroke.dy * direction.vy
    return math.degrees(math.atan2(abs(cross), dot))


def classify_stroke(stroke, direction, min_speed=None, tolerance_deg=None):
    """Decide whether ``stroke`` is a valid cut in ``direction``.

    The speed is checked first: a slow touch is never a *wrong* cut, only too slow.
    Returns ``Verdict(kind, speed, deviation)`` with speed in field widths per second
    and the deviation in degrees.
    """
    if stroke is None:
        return Verdict(CUT_NONE, 0.0, 0.0)
    min_speed = config.CUT_MIN_SPEED if min_speed is None else min_speed
    tolerance_deg = config.CUT_TOLERANCE_DEG if tolerance_deg is None else tolerance_deg
    deviation = angle_deviation(stroke, direction)
    if stroke.speed < min_speed:
        return Verdict(CUT_TOO_SLOW, stroke.speed, deviation)
    kind = CUT_OK if deviation <= tolerance_deg else CUT_WRONG
    return Verdict(kind, stroke.speed, deviation)
