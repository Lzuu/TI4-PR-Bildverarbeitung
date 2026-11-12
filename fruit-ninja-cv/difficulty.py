"""Difficulty presets.

Each level bundles the gameplay/physics values that differ between difficulties.
Values that are the same across all levels (fruit radius, slice animation,
combo rules) stay in ``config.py``.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Difficulty:
    name: str
    gravity: float            # downward acceleration (px/s^2 at REF_HEIGHT)
    launch_vy: tuple          # initial upward velocity range (px/s, negative = up)
    drift_vx: tuple           # horizontal speed range (px/s, direction biased to centre)
    spawn_interval: tuple     # seconds between throws
    burst_weights: tuple      # probabilities of 1 / 2 / 3 fruits per throw
    bomb_probability: float   # chance a thrown object is a bomb
    lives: int                # starting lives


# Lives are always 3; the difficulty is purely about how FAST the fruits fly.
# Higher gravity + launch velocity => faster arcs (same peak height, shorter
# airtime), faster spawning, and a bit more horizontal drift.
# Tuned per frame at 30 fps, converted to seconds (#42, PERF-01):
# gravity x 30^2, velocities x 30, frame intervals / 30.
EASY = Difficulty(
    name="Einfach", gravity=378.0, launch_vy=(-630.0, -510.0), drift_vx=(15.0, 90.0),
    spawn_interval=(1.5, 80 / 30), burst_weights=(0.80, 0.18, 0.02),
    bomb_probability=0.12, lives=3)

MEDIUM = Difficulty(
    name="Mittel", gravity=495.0, launch_vy=(-720.0, -570.0), drift_vx=(30.0, 120.0),
    spawn_interval=(32 / 30, 62 / 30), burst_weights=(0.70, 0.22, 0.08),
    bomb_probability=0.12, lives=3)

HARD = Difficulty(
    name="Schwer", gravity=648.0, launch_vy=(-810.0, -660.0), drift_vx=(45.0, 165.0),
    spawn_interval=(20 / 30, 40 / 30), burst_weights=(0.60, 0.28, 0.12),
    bomb_probability=0.12, lives=3)

# Order corresponds to the menu keys 1 / 2 / 3
LEVELS = [EASY, MEDIUM, HARD]
DEFAULT = MEDIUM
