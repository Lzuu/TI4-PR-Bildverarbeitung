"""Game state: spawning, physics update, collision, score, lives, combos."""

import logging
import random

import config
from fruit import Fruit
from difficulty import DEFAULT
from sound import NullSound
from utils import point_segment_distance, draw_text
from heart import draw_heart

logger = logging.getLogger(__name__)


class Game:
    """Game state; ``update(blade, dt)`` advances it by ``dt`` real seconds.

    Physics, throws and the combo window run in *game time* (``dt`` times
    ``time_scale``, e.g. 0.5 for a slow-motion power-up); the combo banner runs
    in real time. ``rng`` makes throws reproducible in tests.
    """

    def __init__(self, width, height, difficulty=None, sounds=None, rng=random):
        self.width = width
        self.height = height
        self.difficulty = difficulty or DEFAULT
        self.sounds = sounds or NullSound()
        self.rng = rng
        self.time_scale = 1.0
        self.reset()

    def reset(self):
        self.fruits = []
        self.score = 0
        self.lives = self.difficulty.lives
        self.game_over = False
        self._spawn_timer = 0.0
        self._next_spawn = self.rng.uniform(*self.difficulty.spawn_interval)
        # Combo tracking
        self._combo_count = 0
        self._combo_left = 0.0            # game seconds until the open combo closes
        self.combo_text = ""
        self.combo_banner_left = 0.0      # real seconds the banner is still shown

    # -- core update --------------------------------------------------------
    def update(self, blade, dt=1.0 / config.REF_FPS):
        """One camera frame: cut with the current blade, then advance ``dt`` seconds.

        Cutting uses the positions the player saw; afterwards the fruits move
        and new ones are thrown, each already advanced by the part of ``dt``
        after its throw time -- so the result does not depend on the frame rate.
        """
        if self.game_over:
            return
        real_dt = min(max(dt, 0.0), config.MAX_FRAME_DT)
        game_dt = real_dt * self.time_scale

        sliced = self._handle_slicing(blade)
        if self.game_over:                    # a bomb ended the run: nothing moves any more (#64)
            return
        self._update_combo(sliced, game_dt)

        # Advance physics and prune finished / missed fruits
        remaining = []
        for fruit in self.fruits:
            fruit.update(game_dt)
            if fruit.is_finished():
                continue
            if fruit.is_missed(self.height):
                if not fruit.is_bomb:
                    self.lives -= 1          # Missing a bomb is fine
                    logger.info("Missed %s -- lives left: %d",
                                fruit.name, self.lives)
                    self.sounds.play("miss")
                continue
            remaining.append(fruit)
        self.fruits = remaining
        if not self.game_over:
            self._handle_spawn(game_dt)

        if self.combo_banner_left > 0:
            self.combo_banner_left = max(0.0, self.combo_banner_left - real_dt)

        if self.lives <= 0:
            self.lives = 0
            self._end_game()

    def _end_game(self):
        if self._combo_left > 0:
            self._finalize_combo()
        self.game_over = True

    def _handle_spawn(self, dt):
        self._spawn_timer += dt
        while self._spawn_timer >= self._next_spawn:
            self._spawn_timer -= self._next_spawn
            age = self._spawn_timer              # thrown this long before the frame ends
            self._next_spawn = self.rng.uniform(*self.difficulty.spawn_interval)
            # A throw may toss 1-3 fruits at once (a "burst"), like the original
            count = self.rng.choices((1, 2, 3),
                                     weights=self.difficulty.burst_weights)[0]
            for _ in range(count):
                fruit = Fruit(self.width, self.height, self.difficulty, self.rng)
                fruit.update(age)
                self.fruits.append(fruit)

    def _handle_slicing(self, blade):
        """Slice fruits the marker touches. Returns the number of fruits cut."""
        cur = blade.current_point()
        if cur is None:
            return 0
        seg = blade.segment()
        # Check the swipe segment if we have one (so fast passes still register),
        # otherwise fall back to the single current point (a resting touch).
        prev = seg[0] if seg is not None else cur
        cut_angle = blade.angle()
        sliced = 0
        for fruit in self.fruits:
            if fruit.sliced:
                continue
            if point_segment_distance(fruit.center, prev, cur) <= fruit.radius:
                fruit.slice(cut_angle)
                if fruit.is_bomb:
                    self.sounds.play("bomb")
                    logger.info("Sliced a BOMB -- game over at score %d",
                                self.score)
                    if sliced:                # the fruits cut before the bomb belong to the combo
                        self._combo_count += sliced
                        self._combo_left = config.COMBO_WINDOW_SECONDS
                    self._end_game()
                    break                    # no further scoring this frame
                self.score += 1
                sliced += 1
                self.sounds.play("slice")
        return sliced

    # -- combos -------------------------------------------------------------
    def _update_combo(self, sliced, dt=1.0 / config.REF_FPS):
        if self.game_over:
            return
        if sliced > 0:
            self._combo_count += sliced
            self._combo_left = config.COMBO_WINDOW_SECONDS
        elif self._combo_left > 0:
            self._combo_left -= dt
            if self._combo_left <= 1e-9:
                self._finalize_combo()

    def _finalize_combo(self):
        if self._combo_count >= config.COMBO_MIN:
            bonus = self._combo_count * config.COMBO_BONUS_PER_FRUIT
            self.score += bonus
            self.combo_text = f"COMBO x{self._combo_count}!  +{bonus}"
            self.combo_banner_left = config.COMBO_BANNER_SECONDS
            self.sounds.play("combo")
            logger.info("Combo x%d -> +%d (score now %d)",
                        self._combo_count, bonus, self.score)
        self._combo_count = 0
        self._combo_left = 0.0

    # -- drawing ------------------------------------------------------------
    def draw(self, frame):
        for fruit in self.fruits:
            fruit.draw(frame)

    def draw_hud(self, frame):
        s = config.SCALE
        draw_text(frame, f"Score: {self.score}", (int(16 * s), int(38 * s)),
                  scale=0.9)
        draw_text(frame, "Leben", (int(16 * s), int(84 * s)), scale=0.75)
        hsize = int(30 * s)
        hx = int(135 * s)
        for i in range(self.lives):
            draw_heart(frame, (hx + i * int(40 * s), int(78 * s)), hsize)
        draw_text(frame, self.difficulty.name, (int(16 * s), int(120 * s)),
                  scale=0.6, color=(200, 200, 200))

    def draw_combo(self, frame):
        if self.combo_banner_left > 0 and self.combo_text:
            h, w = frame.shape[:2]
            draw_text(frame, self.combo_text, (w // 2, int(h * 0.28)),
                      scale=1.2, color=(0, 215, 255), thickness=3, center=True)
