"""Space background: cut the player out of the camera feed and put them in space.

Classical computer vision, no ML. Pipeline:

  reference (empty background, captured once at runtime)
      + live frame
  -> cv2.absdiff -> grey -> threshold -> morphology (open/close)
  -> foreground mask (the player)
  -> composite: keep the player, replace every other pixel with a space image.

The space image is a static asset (``assets/space.png``). Like the sounds and
the heart sprite, it is *generated once* on first use (a star-field with a soft
nebula) if the file does not exist yet -- so the game needs no external image
file, but you can drop in any real space photo to override it.
"""

import logging
import os

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SPACE_PATH = os.path.join(ASSET_DIR, "space.png")


# --------------------------------------------------------------------------- #
# Segmentation (pure, headless-testable)
# --------------------------------------------------------------------------- #
def foreground_mask(frame, reference, threshold=None, kernel_size=None):
    """Return a uint8 mask (0/255) of pixels that differ from ``reference``.

    ``frame`` and ``reference`` are BGR images of identical shape. A pixel whose
    absolute difference (converted to grey) exceeds ``threshold`` is foreground
    (the player); morphology then removes speckle noise and closes small holes.
    """
    if threshold is None:
        threshold = config.BG_DIFF_THRESHOLD
    if kernel_size is None:
        kernel_size = config.BG_MORPH_KERNEL
    diff = cv2.absdiff(frame, reference)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def composite_space(frame, mask, space_img):
    """Keep ``frame`` where ``mask`` is set (the player), else show ``space_img``.

    ``space_img`` must match the frame size. Returns a *new* BGR image; the
    inputs are not modified.
    """
    mask3 = mask[:, :, None].astype(bool)
    return np.where(mask3, frame, space_img)


# --------------------------------------------------------------------------- #
# Space backdrop asset (generated once if missing, like the sounds)
# --------------------------------------------------------------------------- #
def _generate_space(width, height, seed=7):
    """A star-field with a soft colour nebula -- a believable space backdrop."""
    rng = np.random.default_rng(seed)

    # Dark base with a faint vertical blue -> purple gradient (BGR)
    top = np.array([40, 12, 8], np.float32)
    bottom = np.array([25, 6, 20], np.float32)
    ramp = np.linspace(0, 1, height, dtype=np.float32)[:, None, None]
    img = np.repeat(top * (1 - ramp) + bottom * ramp, width, axis=1)

    # A few soft nebula blobs, heavily blurred
    nebula = np.zeros((height, width, 3), np.float32)
    small = min(width, height)
    for _ in range(5):
        cx, cy = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = int(rng.integers(max(1, small // 6), max(2, small // 3)))
        col = (rng.uniform(0.3, 1.0, 3) * np.array([180, 90, 160], np.float32))
        cv2.circle(nebula, (cx, cy), r, col.tolist(), -1)
    nebula = cv2.GaussianBlur(nebula, (0, 0), small * 0.08)
    img = np.clip(img + nebula * 0.5, 0, 255)

    # Many faint stars (single pixels)
    n_stars = max(1, (width * height) // 900)
    xs = rng.integers(0, width, n_stars)
    ys = rng.integers(0, height, n_stars)
    br = rng.uniform(0.3, 1.0, n_stars)
    img[ys, xs] = np.minimum(255.0, img[ys, xs] + 255.0 * br[:, None])

    # A handful of bright glowing stars
    for _ in range(max(6, n_stars // 200)):
        x, y = int(rng.integers(0, width)), int(rng.integers(0, height))
        glow = np.zeros((height, width, 3), np.float32)
        cv2.circle(glow, (x, y), max(4, small // 120), (255, 240, 220), -1)
        img = np.clip(img + cv2.GaussianBlur(glow, (0, 0), 4) * 0.8, 0, 255)
        cv2.circle(img, (x, y), max(1, small // 360), (255, 255, 255), -1)

    return img.astype(np.uint8)


def load_space(size, path=SPACE_PATH):
    """Load the space backdrop resized to ``size`` = (width, height).

    If the file is missing it is generated once and saved, then loaded -- so the
    game works with no external asset and the file can be replaced by a real
    space photo. Returns ``None`` if the backdrop can neither be loaded nor
    generated (caller then falls back to the raw camera image).
    """
    width, height = size
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, _generate_space(width, height))
            logger.info("Generated space backdrop -> %s", path)
        except Exception:
            logger.warning("Could not generate space backdrop", exc_info=True)
            return None
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        logger.warning("Could not read space backdrop at %s", path)
        return None
    return cv2.resize(img, (width, height))
