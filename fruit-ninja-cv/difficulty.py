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


EASY = Difficulty(
    name="Einfach", gravity=0.42, launch_vy=(-20.0, -17.0), drift_vx=(0.5, 3.0),
    spawn_interval=(55, 95), burst_weights=(0.80, 0.18, 0.02),
    bomb_probability=0.07, lives=5)

MEDIUM = Difficulty(
    name="Mittel", gravity=0.5, launch_vy=(-22.0, -18.0), drift_vx=(1.0, 4.0),
    spawn_interval=(35, 70), burst_weights=(0.70, 0.22, 0.08),
    bomb_probability=0.12, lives=3)

HARD = Difficulty(
    name="Schwer", gravity=0.58, launch_vy=(-24.0, -20.0), drift_vx=(1.5, 5.0),
    spawn_interval=(22, 45), burst_weights=(0.55, 0.30, 0.15),
    bomb_probability=0.18, lives=2)

# Order corresponds to the menu keys 1 / 2 / 3
LEVELS = [EASY, MEDIUM, HARD]
DEFAULT = MEDIUM
