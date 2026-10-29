"""Central configuration for the Fruit-Ninja-CV game.

All tunable constants live here so gameplay, camera and hand-tracking can be
tweaked without touching the logic modules.
"""

# ---------------------------------------------------------------------------
# Camera / window
# ---------------------------------------------------------------------------
CAMERA_INDEX = 1          # 1 = MacBook FaceTime HD. (0 = iPhone Continuity on this Mac.)
                          # Default only -- the camera chosen in the menu wins (SETTINGS_FILE).
CAMERA_MAX_INDEX = 5      # Camera indices 0..4 are searched for the camera selection
SETTINGS_FILE = "settings.json"   # Local settings (chosen camera), not in git
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
# Space background (cut the player out of the camera, show a space backdrop)
# ---------------------------------------------------------------------------
# A reference frame of the *empty* background is captured once (press 'b', or
# the startup countdown). Each live frame is diffed against it to segment the
# player; the rest is replaced by assets/space.png (generated if missing).
BG_ENABLED = True                # Master switch; the menu "Aus" mode disables it per run
BG_MODE = "normal"               # Default mode: "greenscreen" | "normal" (absdiff) | "off"
BG_DIFF_THRESHOLD = 30           # Per-pixel grey diff above which a pixel is "player"
BG_MORPH_KERNEL = 5              # Open/close kernel to clean the foreground mask
BG_MIN_BLOB_FRAC = 0.003         # Drop foreground blobs smaller than this frame fraction
BG_COUNTDOWN_SECONDS = 3.0       # "Step out of frame" countdown before capturing
# Green-screen chroma key. Two detectors are OR-ed so uneven lighting never
# leaves green showing: a generous HSV range AND a lighting-invariant
# "green-dominance" test (green channel exceeds both others by a margin).
GREEN_LOWER_HSV = (30, 35, 25)   # OpenCV HSV: H 0-179, S/V 0-255
GREEN_UPPER_HSV = (95, 255, 255)
GREEN_DOMINANCE = 18             # G must beat max(B, R) by this to count as backdrop
GREEN_SCALE = 1.0                # Full res: ~5 ms/frame; downscaling barely helps but blocks edges
SUIT_ENABLED = True              # Dress the cut-out player in a space suit (suit.py)
SUIT_FACE_DETECT_WIDTH = 320     # Face detection for placing the suit runs at this width

# Background mode "yolo": cut the player out with a YOLOv8-seg person mask
# (ONNX via cv2.dnn, no torch) so no green screen is needed. See yolo_seg.py.
YOLO_ENABLED = True              # Make the "yolo" background mode available
YOLO_INPUT = 320                 # Inference input size (letterboxed square); smaller = faster
YOLO_EVERY_N = 3                 # Infer every Nth frame, reuse the last mask in between (~15 ms/inference)
YOLO_CONF = 0.4                  # Minimum person confidence
YOLO_MASK_THRESH = 0.5           # Prototype-mask threshold

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
PEN_POINT = "tip"                # Tracked pen point: "tip" (upper end) or "centroid"
SABER_ENABLED = True             # Draw the pen as a lightsaber (lightsaber.py)
SABER_LENGTH = 110               # Blade length in px at REF_HEIGHT (scaled), down from the pen point
SABER_HILT_LENGTH = 45           # Hilt length in px at REF_HEIGHT (scaled), below the blade
SABER_HAND_RADIUS = 220          # Hand (skin blob) is searched within this distance of the pen tip

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

# ---------------------------------------------------------------------------
# Player identification by face (YuNet detection + SFace recognition, see face_id.py)
# ---------------------------------------------------------------------------
FACES_DIR = "faces"              # Reference images: faces/<Name>/*.jpg|png (python enroll.py)
FACE_DETECT_SCORE = 0.8          # Minimum YuNet confidence for a face
FACE_MATCH_THRESHOLD = 0.363     # Cosine similarity for "same person" (SFace recommendation)
FACE_INPUT_WIDTH = 640           # Frames are downscaled to this width before detection
FACE_VOTES_NEEDED = 5            # Frames recognising the same player needed to accept
FACE_SCAN_CROPS = 8              # Face crops recorded for a new player
FACE_SCAN_INTERVAL = 0.3         # Seconds between two recorded crops (head may move a bit)
FACE_SCAN_RADIUS = 150           # Radius of the "face here" circle (scaled with SCALE)
FACE_GREETING_SECONDS = 1.5      # How long "Hallo, <Name>!" is shown
KEYBOARD_DWELL_SECONDS = 0.8     # Dwell per key on the on-screen keyboard (faster than menus)
NAME_MAX_LENGTH = 16             # Maximum length of a player name

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
