"""Central configuration for the Fruit-Ninja-CV game.

All tunable constants live here so gameplay, camera and hand-tracking can be
tweaked without touching the logic modules.
"""

# ---------------------------------------------------------------------------
# Camera / window
# ---------------------------------------------------------------------------
CAMERA_INDEX = 1          # 1 = MacBook FaceTime HD. (0 = iPhone Continuity on this Mac.)
WIDTH = 960               # Processing/display width  (frame is resized to this)
HEIGHT = 540              # Processing/display height
MIRROR = True             # Flip horizontally so moving right moves the hand right
WINDOW_NAME = "Fruit Ninja CV"

# ---------------------------------------------------------------------------
# Marker tracking (pink pen used as the "sword", HSV colour segmentation)
# ---------------------------------------------------------------------------
# OpenCV HSV ranges: H 0-179, S 0-255, V 0-255.
# Default range for a vivid pink/magenta marker. The HUE is the key discriminator;
# saturation/brightness only need to clear a floor so lighting matters less.
MARKER_HSV_LOWER = (140, 55, 90)
MARKER_HSV_UPPER = (175, 255, 255)
# Calibration measures the dominant hue and builds the range as:
#   lower = (hue - H_TOLERANCE, S_FLOOR, V_FLOOR)
#   upper = (hue + H_TOLERANCE, 255, 255)
H_TOLERANCE = 10                 # +/- hue window around the sampled colour
S_FLOOR = 55                     # Minimum saturation (ignores the pale white wall)
V_FLOOR = 80                     # Minimum brightness
MIN_MARKER_AREA = 200            # Ignore contours smaller than this (noise)
MORPH_KERNEL = 5                 # Kernel size for open/close morphology
CALIB_BOX = (0.42, 0.32, 0.16, 0.36)  # ROI fractions (x, y, w, h); hold the pen tip here
CALIB_PATCH = 8                  # Half-size (px) of the patch sampled on a click

# ---------------------------------------------------------------------------
# Blade / slicing
# ---------------------------------------------------------------------------
TRAIL_LENGTH = 12                # How many recent marker points form the blade
BLADE_COLOR = (255, 255, 255)    # BGR
BLADE_GLOW_COLOR = (255, 230, 120)
BLADE_MAX_THICKNESS = 10

# ---------------------------------------------------------------------------
# Gameplay / physics (pixels per frame)
# ---------------------------------------------------------------------------
# Fruits are thrown UP from below the bottom edge: they start with an upward
# velocity, gravity decelerates them, they peak and fall back down (projectile
# arc). An un-sliced fruit is only "missed" once it falls back below the bottom.
#
# Difficulty-dependent values (gravity, launch/drift velocity, spawn rate, bomb
# chance, lives) live in difficulty.py. The constants below are shared by all
# difficulty levels.
FRUIT_RADIUS = (34, 52)          # Radius range
SLICE_ANIM_FRAMES = 25           # How long the two halves fly before removal

# ---------------------------------------------------------------------------
# Combos (Fruit-Ninja-style: several fruits within one swipe / short window)
# ---------------------------------------------------------------------------
COMBO_WINDOW_FRAMES = 12         # Slices within this window belong to one combo
COMBO_MIN = 3                    # Minimum fruits for a combo bonus
COMBO_BONUS_PER_FRUIT = 1        # Extra points per fruit in a qualifying combo
COMBO_BANNER_FRAMES = 35         # How long the combo banner stays on screen

# ---------------------------------------------------------------------------
# Highscore
# ---------------------------------------------------------------------------
HIGHSCORE_FILE = "highscores.json"

# Fruit types: (name, BGR fill color)
FRUIT_TYPES = [
    ("apple",      (40, 40, 220)),
    ("orange",     (0, 140, 255)),
    ("lime",       (60, 210, 90)),
    ("blueberry",  (200, 90, 60)),
    ("lemon",      (40, 230, 230)),
]
BOMB_COLOR = (45, 45, 45)

# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------
HUD_COLOR = (255, 255, 255)
HUD_SHADOW = (0, 0, 0)
