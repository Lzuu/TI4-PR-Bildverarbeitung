"""Game state: spawning, physics update, collision, score and lives."""

import logging
import random

import config
from fruit import Fruit
from utils import point_segment_distance, draw_text

logger = logging.getLogger(__name__)


class Game:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.reset()

    def reset(self):
        self.fruits = []
        self.score = 0
        self.lives = config.START_LIVES
        self.game_over = False
        self._spawn_timer = 0
        self._next_spawn = random.randint(*config.SPAWN_INTERVAL)

    # -- core update --------------------------------------------------------
    def update(self, blade):
        if self.game_over:
            return

        self._handle_spawn()
        self._handle_slicing(blade)

        # Advance physics and prune finished / missed fruits
        remaining = []
        for fruit in self.fruits:
            fruit.update()
            if fruit.is_finished():
                continue
            if fruit.is_missed(self.height):
                if not fruit.is_bomb:
                    self.lives -= 1          # Missing a bomb is fine
                    logger.info("Missed %s -- lives left: %d",
                                fruit.name, self.lives)
                continue
            remaining.append(fruit)
        self.fruits = remaining

        if self.lives <= 0:
            self.lives = 0
            self.game_over = True

    def _handle_spawn(self):
        self._spawn_timer += 1
        if self._spawn_timer >= self._next_spawn:
            self._spawn_timer = 0
            self._next_spawn = random.randint(*config.SPAWN_INTERVAL)
            # A throw may toss 1-3 fruits at once (a "burst"), like the original
            count = random.choices((1, 2, 3), weights=config.BURST_WEIGHTS)[0]
            for _ in range(count):
                self.fruits.append(Fruit(self.width, self.height))

    def _handle_slicing(self, blade):
        # Touching a fruit with the marker is enough -- no minimum speed required.
        cur = blade.current_point()
        if cur is None:
            return
        seg = blade.segment()
        # Check the swipe segment if we have one (so fast passes still register),
        # otherwise fall back to the single current point (a resting touch).
        prev = seg[0] if seg is not None else cur
        cut_angle = blade.angle()
        for fruit in self.fruits:
            if fruit.sliced:
                continue
            if point_segment_distance(fruit.center, prev, cur) <= fruit.radius:
                fruit.slice(cut_angle)
                if fruit.is_bomb:
                    self.game_over = True   # Slicing a bomb ends the run
                    logger.info("Sliced a BOMB -- game over at score %d",
                                self.score)
                else:
                    self.score += 1

    # -- drawing ------------------------------------------------------------
    def draw(self, frame):
        for fruit in self.fruits:
            fruit.draw(frame)

    def draw_hud(self, frame):
        draw_text(frame, f"Score: {self.score}", (16, 36), scale=0.9)
        hearts = "<3 " * self.lives
        draw_text(frame, f"Lives: {hearts.strip()}", (16, 72), scale=0.9,
                  color=(80, 80, 255))
