"""Central configuration for the Fruit-Ninja-CV game.

All tunable constants live here so gameplay, camera and hand-tracking can be
tweaked without touching the logic modules.
"""

# ---------------------------------------------------------------------------
# Camera / window
# ---------------------------------------------------------------------------
CAMERA_INDEX = 1          # 1 = MacBook FaceTime HD. (0 = iPhone Continuity on this Mac.)
# WIDTH/HEIGHT are the render canvas size. They default to 960x540 but are set to
# the actual screen resolution at startup (fullscreen, responsive layout). SCALE is
# derived from HEIGHT so physics and UI scale with the resolution.
WIDTH = 960               # Render/display width  (overridden by screen size at runtime)
HEIGHT = 540              # Render/display height (overridden by screen size at runtime)
REF_HEIGHT = 540          # Reference height the gameplay constants were tuned for
SCALE = 1.0               # = HEIGHT / REF_HEIGHT, set at runtime
# Laptop screens (MacBook) expose a ~16:10 usable fullscreen area. We render to
# this aspect and crop the 16:9 webcam to it, so the window fills the screen
# (no grey bar) without distorting the video.
ASPECT = 16 / 10
FULLSCREEN = True         # Open the window in fullscreen
MIRROR = True             # Flip horizontally so moving right moves the hand right
WINDOW_NAME = "Fruit Ninja CV"
MAX_FRAME_FAILURES = 60   # Abort only after this many *consecutive* failed reads

# ---------------------------------------------------------------------------
# Marker tracking (fixed colour ranges -- no self-calibration; see colors.py)
# ---------------------------------------------------------------------------
MIN_MARKER_AREA = 300            # Ignore contours smaller than this (noise/false blobs)
MORPH_KERNEL = 5                 # Kernel size for open/close morphology

# ---------------------------------------------------------------------------
# Selection (VR-style dwell: hold the pen in a circle to confirm)
# ---------------------------------------------------------------------------
DWELL_SECONDS = 3.0              # How long to hold on a target to select it
SELECT_RADIUS = 72               # Radius of the difficulty-selection circles (px)
CONFIRM_RADIUS = 95              # Radius of the central "hold here to start" circle

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
