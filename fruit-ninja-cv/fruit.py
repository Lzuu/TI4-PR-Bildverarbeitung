"""Fruit (and bomb) entities: projectile physics + procedural drawing.

Fruits are thrown up from below the bottom edge with an upward velocity, arc
under gravity and fall back down (like someone tossing them from below). They
are drawn procedurally (filled circle + highlight + stem) so the game runs with
zero external assets. When sliced, a fruit splits into two half-discs that fly
apart perpendicular to the blade direction.
"""

import math
import random

import cv2

import config


class Fruit:
    def __init__(self, width, height):
        self.radius = random.randint(*config.FRUIT_RADIUS)
        # Launch from just below the bottom edge at a random horizontal position
        margin = int(0.12 * width)
        self.x = float(random.randint(margin, width - margin))
        self.y = float(height + self.radius)
        # Arc inward (toward the centre) so fruits stay on screen; throw upward
        direction = 1.0 if self.x < width / 2.0 else -1.0
        self.vx = direction * random.uniform(*config.FRUIT_DRIFT_VX)
        self.vy = random.uniform(*config.FRUIT_LAUNCH_VY)   # negative => upward

        self.is_bomb = random.random() < config.BOMB_PROBABILITY
        if self.is_bomb:
            self.color = config.BOMB_COLOR
            self.name = "bomb"
        else:
            self.name, self.color = random.choice(config.FRUIT_TYPES)

        self.sliced = False
        self.slice_timer = 0
        self.cut_angle = 0.0       # blade direction in radians (for the cut line)

    # -- state --------------------------------------------------------------
    @property
    def center(self):
        return (int(self.x), int(self.y))

    def slice(self, cut_angle):
        self.sliced = True
        self.cut_angle = cut_angle

    def update(self):
        if self.sliced:
            self.slice_timer += 1
        self.x += self.vx
        self.y += self.vy
        self.vy += config.GRAVITY

    def is_missed(self, height):
        """True once an un-sliced fruit falls back out of the bottom.

        Guarded by ``vy > 0`` so a fruit that is still rising just after launch
        (and thus momentarily below the edge) does not count as missed.
        """
        return (not self.sliced) and self.vy > 0 and (self.y - self.radius > height)

    def is_finished(self):
        """True once the slice animation has run its course (remove it)."""
        return self.sliced and self.slice_timer >= config.SLICE_ANIM_FRAMES

    # -- drawing ------------------------------------------------------------
    def draw(self, frame):
        if self.sliced:
            self._draw_halves(frame)
        elif self.is_bomb:
            self._draw_bomb(frame)
        else:
            self._draw_whole(frame, self.center, self.radius)

    def _draw_whole(self, frame, center, radius):
        cx, cy = center
        cv2.circle(frame, (cx, cy), radius, self.color, -1, cv2.LINE_AA)
        # Lighter highlight towards the top-left for a glossy look
        highlight = tuple(min(255, c + 70) for c in self.color)
        cv2.circle(frame, (cx - radius // 3, cy - radius // 3),
                   max(3, radius // 4), highlight, -1, cv2.LINE_AA)
        # Small brown stem
        cv2.line(frame, (cx, cy - radius), (cx, cy - radius - 10),
                 (30, 70, 110), 3, cv2.LINE_AA)

    def _draw_bomb(self, frame):
        cx, cy = self.center
        cv2.circle(frame, (cx, cy), self.radius, self.color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), self.radius, (90, 90, 90), 2, cv2.LINE_AA)
        # Fuse + spark
        cv2.line(frame, (cx, cy - self.radius), (cx + 10, cy - self.radius - 14),
                 (60, 80, 110), 3, cv2.LINE_AA)
        cv2.circle(frame, (cx + 10, cy - self.radius - 14), 4,
                   (0, 200, 255), -1, cv2.LINE_AA)
        # Highlight
        cv2.circle(frame, (cx - self.radius // 3, cy - self.radius // 3),
                   max(3, self.radius // 5), (110, 110, 110), -1, cv2.LINE_AA)

    def _draw_halves(self, frame):
        """Two half-discs separating perpendicular to the cut."""
        spread = self.slice_timer * 3
        # Perpendicular to the blade direction
        perp = self.cut_angle + math.pi / 2
        ox = math.cos(perp) * spread
        oy = math.sin(perp) * spread
        angle_deg = math.degrees(self.cut_angle)
        axes = (self.radius, self.radius)

        c1 = (int(self.x + ox), int(self.y + oy))
        c2 = (int(self.x - ox), int(self.y - oy))
        if self.is_bomb:
            # A sliced bomb still shows, but the game is already over
            cv2.circle(frame, self.center, self.radius, self.color, -1, cv2.LINE_AA)
            return
        cv2.ellipse(frame, c1, axes, angle_deg, 0, 180, self.color, -1, cv2.LINE_AA)
        cv2.ellipse(frame, c2, axes, angle_deg, 180, 360, self.color, -1, cv2.LINE_AA)
        # Inner "flesh" line on each half
        lighter = tuple(min(255, c + 50) for c in self.color)
        cv2.ellipse(frame, c1, (self.radius - 6, self.radius - 6), angle_deg,
                    0, 180, lighter, 3, cv2.LINE_AA)
        cv2.ellipse(frame, c2, (self.radius - 6, self.radius - 6), angle_deg,
                    180, 360, lighter, 3, cv2.LINE_AA)
