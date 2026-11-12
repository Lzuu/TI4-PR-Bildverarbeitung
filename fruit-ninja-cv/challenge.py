"""Challenge mode: direction fruits that need a fast cut in the arrow's direction (#33).

A direction fruit carries an arrow and a double ring (colour alone is not the cue).
It is cut only if

1. the pen's path between two consecutive valid positions touches the fruit (so a fast
   move cannot jump through it between two frames),
2. the stroke is fast enough (``CUT_MIN_SPEED`` field widths per second), and
3. its direction is within ``CUT_TOLERANCE_DEG`` of the arrow.

The analysis itself lives in ``motion.py``; this module only turns its verdicts into game
rules. A cut in the wrong direction leaves the fruit in place and shows a short notice; a
slow touch or standing still is no cut and no error. Plain fruits and bombs keep the
classic touch rules, missed fruits and bombs the known life rules. The challenge lasts
``CHALLENGE_SECONDS`` of real time or until all lives are gone.
"""

import logging
import math
import random

import cv2

import config
from fruit import Fruit
from game import Game
from motion import CUT_OK, CUT_TOO_SLOW, CUT_WRONG, DIRECTIONS, classify_stroke
from utils import draw_text, point_segment_distance

logger = logging.getLogger(__name__)


class Feedback:
    """A short notice on screen: ``kind`` is "good", "wrong" or "slow"."""

    def __init__(self, kind, text, pos, seconds):
        self.kind, self.text, self.pos = kind, text, pos
        self.total = self.left = seconds


class DirectionFruit(Fruit):
    """A fruit that can only be cut in ``direction`` (see ``motion.DIRECTIONS``)."""

    def __init__(self, width, height, direction, diff=None, rng=random):
        super().__init__(width, height, diff, rng)
        self.is_bomb = False
        self.direction = direction
        self.report_at = 0.0               # earliest time the next rejected cut is reported

    def draw(self, frame):
        super().draw(frame)
        if self.sliced:
            return                         # the halves fly apart without the symbol
        s = config.SCALE
        cx, cy = self.center
        r = self.radius
        white = (255, 255, 255)
        # double ring: the symbol that marks a direction fruit (not only its colour)
        cv2.circle(frame, (cx, cy), r + int(7 * s), white, max(2, int(3 * s)), cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), r + int(2 * s), (30, 30, 30), max(1, int(2 * s)), cv2.LINE_AA)
        # arrow on a dark disc
        cv2.circle(frame, (cx, cy), int(r * 0.62), (35, 35, 35), -1, cv2.LINE_AA)
        length = r * 0.5
        a = (int(cx - self.direction.vx * length), int(cy - self.direction.vy * length))
        b = (int(cx + self.direction.vx * length), int(cy + self.direction.vy * length))
        cv2.arrowedLine(frame, a, b, white, max(3, int(5 * s)), cv2.LINE_AA, tipLength=0.5)


class ChallengeGame(Game):
    def __init__(self, width, height, difficulty=None, sounds=None, rng=random, seconds=None):
        self.seconds = config.CHALLENGE_SECONDS if seconds is None else seconds
        self._motion = None
        super().__init__(width, height, difficulty, sounds, rng)

    def reset(self):
        super().reset()
        self.time_left = float(self.seconds)
        self.time_up = False
        self.directions_done = 0           # correctly cut direction fruits
        self.wrong_cuts = 0
        self.feedback = []
        self.elapsed = 0.0                 # real seconds since the start
        self._bag = []                     # shuffled directions: every one comes up evenly
        logger.info("Challenge started: %s, %.0f s, %d lives", self.difficulty.name,
                    self.seconds, self.lives)

    # -- what the screens show ---------------------------------------------
    @property
    def board_name(self):
        return f"Challenge {self.difficulty.name}"

    @property
    def title(self):
        return "ZEIT ABGELAUFEN" if self.time_up else "GAME OVER"

    def result_text(self):
        return f"{self.score}  |  Richtungsaufgaben: {self.directions_done}"

    # -- throwing -------------------------------------------------------------
    def _next_direction(self):
        if not self._bag:
            self._bag = list(DIRECTIONS)
            self.rng.shuffle(self._bag)
        return self._bag.pop()

    def _new_fruit(self):
        fruit = super()._new_fruit()
        if fruit.is_bomb or self.rng.random() >= config.CHALLENGE_DIRECTION_SHARE:
            return fruit
        directional = DirectionFruit(self.width, self.height, self._next_direction(),
                                     self.difficulty, self.rng)
        directional.x, directional.y = fruit.x, fruit.y      # same throw, other kind
        directional.vx, directional.vy = fruit.vx, fruit.vy
        directional.radius = fruit.radius
        directional.name, directional.color = fruit.name, fruit.color
        return directional

    # -- the update ------------------------------------------------------------
    def update(self, blade, dt=1.0 / config.REF_FPS, motion=None):
        if self.game_over:
            return
        real_dt = min(max(dt, 0.0), config.MAX_FRAME_DT)
        self._motion = motion
        self.elapsed += real_dt
        super().update(blade, dt, motion)
        for note in self.feedback:
            note.left -= real_dt
        self.feedback = [n for n in self.feedback if n.left > 0]
        if not self.game_over:
            self.time_left = max(0.0, self.time_left - real_dt)    # real time, not game time
            if self.time_left <= 0:
                self.time_up = True
                self._end_game()
                logger.info("Challenge over: time is up -- score %d, %d direction cuts, "
                            "%d wrong cuts", self.score, self.directions_done, self.wrong_cuts)

    def _touch_cuttable(self, fruit):
        return getattr(fruit, "direction", None) is None

    def _handle_slicing(self, blade):
        sliced = super()._handle_slicing(blade)     # plain fruits and bombs, by touch
        if self.game_over or self._motion is None:
            return sliced
        seg = self._motion.segment()
        if seg is None:
            return sliced                           # pen new or lost: nothing to cut with
        prev, cur = seg
        stroke = self._motion.stroke()
        for fruit in self.fruits:
            direction = getattr(fruit, "direction", None)
            if direction is None or fruit.sliced:
                continue
            if point_segment_distance(fruit.center, prev, cur) > fruit.radius:
                continue
            verdict = classify_stroke(stroke, direction)
            if verdict.kind == CUT_OK:
                fruit.slice(stroke.angle)
                self.score += config.CHALLENGE_POINTS
                self.directions_done += 1
                sliced += 1
                self.sounds.play("slice")
                self.feedback_add("good", f"+{config.CHALLENGE_POINTS}", fruit.center)
                logger.info("Cut ok: direction=%s speed=%.2f deviation=%.0f",
                            direction.name, verdict.speed, verdict.deviation)
            elif self.elapsed >= fruit.report_at:
                if verdict.kind == CUT_WRONG:
                    fruit.report_at = self.elapsed + config.CHALLENGE_REPORT_COOLDOWN
                    self.wrong_cuts += 1
                    self.feedback_add("wrong", "Falsche Richtung", fruit.center)
                    logger.info("Cut wrong_direction: direction=%s speed=%.2f deviation=%.0f",
                                direction.name, verdict.speed, verdict.deviation)
                elif (verdict.kind == CUT_TOO_SLOW
                      and verdict.speed >= config.CUT_HINT_FRACTION * config.CUT_MIN_SPEED):
                    # a real attempt that was too slow gets a hint; resting is no error
                    fruit.report_at = self.elapsed + config.CHALLENGE_REPORT_COOLDOWN
                    self.feedback_add("slow", "Schneller!", fruit.center)
                    logger.info("Cut too_slow: direction=%s speed=%.2f deviation=%.0f",
                                direction.name, verdict.speed, verdict.deviation)
        return sliced

    # -- feedback and drawing ---------------------------------------------------
    def feedback_add(self, kind, text, pos):
        self.feedback.append(Feedback(kind, text, pos, config.CHALLENGE_FEEDBACK_SECONDS))

    FEEDBACK_COLORS = {"good": (80, 220, 80), "wrong": (60, 60, 255), "slow": (0, 170, 255)}

    def draw_feedback(self, frame):
        s = config.SCALE
        for note in self.feedback:
            rise = (1.0 - note.left / note.total) * 40 * s
            x, y = note.pos
            draw_text(frame, note.text, (x, y - 60 * s - rise), scale=0.9,
                      color=self.FEEDBACK_COLORS[note.kind], thickness=3, center=True)

    def draw_hud(self, frame):
        super().draw_hud(frame)
        s = config.SCALE
        w = frame.shape[1]
        left = math.ceil(self.time_left)
        draw_text(frame, f"Zeit: {left}", (w // 2, int(38 * s)), scale=1.1, center=True,
                  color=(80, 80, 255) if left <= 10 else config.HUD_COLOR, thickness=3)
        draw_text(frame, f"Richtungen: {self.directions_done}", (int(16 * s), int(150 * s)),
                  scale=0.6, color=(200, 200, 200))
