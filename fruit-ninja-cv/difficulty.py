"""Difficulty presets.

Each level bundles the gameplay/physics values that differ between difficulties.
Values that are the same across all levels (fruit radius, slice animation,
combo rules) stay in ``config.py``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Difficulty:
    name: str
    gravity: float            # downward acceleration (px/frame^2)
    launch_vy: tuple          # initial upward velocity range (negative = up)
    drift_vx: tuple           # horizontal speed range (direction biased to centre)
    spawn_interval: tuple     # frames between throws
    burst_weights: tuple      # probabilities of 1 / 2 / 3 fruits per throw
    bomb_probability: float   # chance a thrown object is a bomb
    lives: int                # starting lives


# Lives are always 3; the difficulty is purely about how FAST the fruits fly.
# Higher gravity + launch velocity => faster arcs (same peak height, shorter
# airtime), faster spawning, and a bit more horizontal drift.
EASY = Difficulty(
    name="Einfach", gravity=0.42, launch_vy=(-21.0, -17.0), drift_vx=(0.5, 3.0),
    spawn_interval=(45, 80), burst_weights=(0.80, 0.18, 0.02),
    bomb_probability=0.12, lives=3)

MEDIUM = Difficulty(
    name="Mittel", gravity=0.55, launch_vy=(-24.0, -19.0), drift_vx=(1.0, 4.0),
    spawn_interval=(32, 62), burst_weights=(0.70, 0.22, 0.08),
    bomb_probability=0.12, lives=3)

HARD = Difficulty(
    name="Schwer", gravity=0.72, launch_vy=(-27.0, -22.0), drift_vx=(1.5, 5.5),
    spawn_interval=(20, 40), burst_weights=(0.60, 0.28, 0.12),
    bomb_probability=0.12, lives=3)

# Order corresponds to the menu keys 1 / 2 / 3
LEVELS = [EASY, MEDIUM, HARD]
DEFAULT = MEDIUM
