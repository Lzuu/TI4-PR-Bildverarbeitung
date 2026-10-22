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

# Background modes the player cycles through in the menu. ``greenscreen`` keys a
# green backdrop out by colour; ``normal`` uses the empty-background reference
# (press 'b'); ``off`` shows the raw camera image (no space compositing).
MODE_GREENSCREEN = "greenscreen"
MODE_NORMAL = "normal"
MODE_OFF = "off"
MODES = (MODE_GREENSCREEN, MODE_NORMAL, MODE_OFF)
MODE_LABELS = {
    MODE_GREENSCREEN: "Greenscreen",
    MODE_NORMAL: "Normal",
    MODE_OFF: "Aus",
}


def next_mode(mode):
    """The next background mode in the cycle (wraps around)."""
    i = MODES.index(mode) if mode in MODES else -1
    return MODES[(i + 1) % len(MODES)]


# --------------------------------------------------------------------------- #
# Segmentation (pure, headless-testable)
# --------------------------------------------------------------------------- #
def _fill_silhouette(mask, min_area):
    """Turn a speckly mask into solid filled blobs.

    External contours above ``min_area`` are redrawn *filled*, so interior holes
    (e.g. low-contrast face regions that fell under the threshold) become part of
    the silhouette, and small noise blobs are dropped. Returns a new mask.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(mask)
    keep = [c for c in contours if cv2.contourArea(c) >= min_area]
    if keep:
        cv2.drawContours(out, keep, -1, 255, thickness=cv2.FILLED)
    return out


def _drop_small_blobs(mask, min_area):
    """Zero out foreground connected components smaller than ``min_area``.

    Unlike ``_fill_silhouette`` this keeps interior holes -- background enclosed
    by the player, e.g. the gap inside looped arms -- as background, instead of
    filling the player's outer contour solid. Returns a new mask.
    """
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    lut = np.zeros(n, np.uint8)                 # label -> 255 if big enough, else 0
    lut[1:] = np.where(stats[1:, cv2.CC_STAT_AREA] >= min_area, 255, 0)
    return lut[labels]                          # fast per-pixel relabel (LUT index)


def _refine_mask(mask, kernel_size, fill_holes, min_blob_frac):
    """Clean a raw 0/255 foreground mask: open/close morphology, then optionally
    fill the silhouette so interior holes close and small noise blobs drop."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    if fill_holes:
        mh, mw = mask.shape[:2]
        mask = _fill_silhouette(mask, min_blob_frac * mh * mw)
    return mask


def foreground_mask(frame, reference, threshold=None, kernel_size=None, scale=1.0,
                    fill_holes=True, min_blob_frac=None):
    """Return a uint8 mask (0/255) of pixels that differ from ``reference``.

    ``frame`` and ``reference`` are BGR images of identical shape. A pixel whose
    absolute difference (converted to grey) exceeds ``threshold`` is foreground
    (the player); morphology then removes speckle noise and closes small holes.

    ``scale`` < 1.0 runs the whole segmentation on a downscaled copy (much
    cheaper morphology) and upscales the resulting mask back to the full frame
    size -- the dominant per-frame cost of the space background.

    ``fill_holes`` redraws the external contours filled, so interior holes (a
    face that blends into the wall) are closed and the player becomes a solid
    silhouette; blobs below ``min_blob_frac`` of the (downscaled) area are
    dropped.
    """
    if threshold is None:
        threshold = config.BG_DIFF_THRESHOLD
    if kernel_size is None:
        kernel_size = config.BG_MORPH_KERNEL
    if min_blob_frac is None:
        min_blob_frac = config.BG_MIN_BLOB_FRAC

    h, w = frame.shape[:2]
    if scale != 1.0:
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)
        reference = cv2.resize(reference, (sw, sh), interpolation=cv2.INTER_AREA)

    diff = cv2.absdiff(frame, reference)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    mask = _refine_mask(mask, kernel_size, fill_holes, min_blob_frac)

    if scale != 1.0:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def _green_backdrop(frame, lower, upper, dominance):
    """Mask (0/255) of green-screen pixels, robust to uneven lighting.

    OR of two cheap, all-C detectors so no shade of green slips through:
      * HSV ``inRange`` -- catches the green hue band;
      * green-dominance -- ``G - max(B, R) >= dominance`` (saturating subtract),
        which is brightness-invariant and so still fires on dark/shadowed or
        washed-out green that falls outside the HSV box.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    b, g, r = cv2.split(frame)
    greenness = cv2.subtract(g, cv2.max(b, r))      # 0 where green is not dominant
    _, dom = cv2.threshold(greenness, dominance, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_or(green, dom)


def greenscreen_mask(frame, lower=None, upper=None, kernel_size=None, scale=1.0,
                     drop_small=True, min_blob_frac=None, dominance=None):
    """Return a uint8 mask (0/255) of the player in front of a green screen.

    Chroma key: everything detected as the green backdrop becomes background,
    everything else is the player. Same output contract as ``foreground_mask``
    (player = 255) so the caller composites it identically, and no empty
    reference is needed. The backdrop is cleaned with a morphological *close*
    first so stray non-green speckles inside the green field are absorbed (the
    field is then fully keyed out, not just its edges); the player mask is then
    opened and, with ``drop_small``, tiny noise components are discarded. Unlike
    the absdiff path the silhouette is **not** filled -- green enclosed by the
    player (looped arms) must stay keyed out, so genuine holes are preserved.
    ``scale`` < 1 runs the whole segmentation on a downscaled copy (cheaper) and
    upscales the mask back -- the per-frame cost of the green-screen mode.
    """
    if lower is None:
        lower = config.GREEN_LOWER_HSV
    if upper is None:
        upper = config.GREEN_UPPER_HSV
    if kernel_size is None:
        kernel_size = config.BG_MORPH_KERNEL
    if min_blob_frac is None:
        min_blob_frac = config.BG_MIN_BLOB_FRAC
    if dominance is None:
        dominance = config.GREEN_DOMINANCE

    h, w = frame.shape[:2]
    if scale != 1.0:
        sw, sh = max(1, int(w * scale)), max(1, int(h * scale))
        frame = cv2.resize(frame, (sw, sh), interpolation=cv2.INTER_AREA)

    green = _green_backdrop(frame, lower, upper, dominance)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, kernel)   # solidify the backdrop
    mask = cv2.bitwise_not(green)                              # player = not backdrop
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)      # drop stray foreground specks
    if drop_small:
        mh, mw = mask.shape[:2]
        mask = _drop_small_blobs(mask, min_blob_frac * mh * mw)

    if scale != 1.0:
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask


def composite_space(frame, mask, space_img):
    """Show ``space_img`` wherever ``mask`` is 0, keep ``frame`` where it is set.

    Uses ``cv2.copyTo`` (OpenCV's optimised masked copy) to write the backdrop
    into ``frame`` *in place* -- far cheaper than ``np.where`` on a full frame,
    which is the per-frame hot path. ``space_img`` must match the frame size.
    """
    cv2.copyTo(space_img, cv2.bitwise_not(mask), frame)
    return frame


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
