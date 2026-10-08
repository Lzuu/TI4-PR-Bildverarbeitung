"""Fixed pen colours (no self-calibration).

Each colour is a fixed HSV range plus a BGR value used to draw its UI circle.
Hue is the main discriminator; saturation/brightness floors keep skin and the
pale background out. Pink is the default.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PenColor:
    name: str
    lower: tuple      # HSV lower bound
    upper: tuple      # HSV upper bound
    display: tuple    # BGR colour for the UI


PINK = PenColor("Pink", (140, 70, 90), (175, 255, 255), (180, 0, 255))
GELB = PenColor("Gelb", (22, 90, 120), (38, 255, 255), (0, 230, 230))
GRUEN = PenColor("Gruen", (40, 70, 70), (85, 255, 255), (60, 200, 90))

COLORS = [PINK, GELB, GRUEN]
DEFAULT = PINK
