"""Space suit: dress the cut-out player in a suit (on top of the space background).

The suit is not a static picture: the player's own silhouette is turned into
suit fabric, so it moves with the arms and the body. Per frame (only while the
space background is active):

  foreground mask (background.py)
  -> suit region = silhouette minus head (around the face), hands (skin colour)
     and the pen
  -> fabric: light grey/white, darker towards the silhouette edge (rounded,
     tube-like arms) and keeping the folds of the real clothes
  -> details on top (collar ring, chest panel, patch) anchored at the face.

Classical image processing only (masks, distance transform, blur) -- a pose
network would follow the arms too, but costs ~100-200 ms per frame on a laptop.

The details sprite is generated once into ``assets/suit_details.png`` (BGRA) if it is
missing; replace the file with real artwork to change the look.
"""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SUIT_PATH = os.path.join(ASSET_DIR, "suit_details.png")   # new name: old torso sprite is stale

SUIT_WIDTH_PER_FACE = 3.2        # suit width = face width * this
SPRITE_W, SPRITE_H = 400, 480    # generated sprite size (aspect used for placement)
MIN_PLAYER_FRAC = 0.01           # blobs smaller than this frame fraction are noise


# --------------------------------------------------------------------------- #
# Player position (pure)
# --------------------------------------------------------------------------- #
def _largest_blob(mask, min_frac):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    blob = max(contours, key=cv2.contourArea)
    if cv2.contourArea(blob) < min_frac * mask.shape[0] * mask.shape[1]:
        return None
    return blob


def player_bbox(mask, min_frac=MIN_PLAYER_FRAC):
    """Bounding box (x, y, w, h) of the player (largest foreground blob) or None."""
    blob = _largest_blob(mask, min_frac)
    return None if blob is None else tuple(int(v) for v in cv2.boundingRect(blob))


def player_centroid(mask, min_frac=MIN_PLAYER_FRAC):
    """Centre of mass (x, y) of the player or None."""
    blob = _largest_blob(mask, min_frac)
    if blob is None:
        return None
    m = cv2.moments(blob)
    if m["m00"] == 0:
        return None
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


def head_mask(shape, face=None, bbox=None):
    """Mask of the head (kept free of fabric).

    With a face: an ellipse around face and neck plus the area above it (hair).
    Without a face: the top 30 % of the player's bounding box.
    """
    head = np.zeros(shape[:2], np.uint8)
    if face is not None:
        fx, fy, fw, fh = face
        center = (int(fx + fw / 2), int(fy + fh * 0.45))
        cv2.ellipse(head, center, (int(fw * 0.75), int(fh * 0.85)), 0, 0, 360, 255, -1)
        cv2.rectangle(head, (int(fx - fw * 0.15), 0), (int(fx + fw * 1.15), int(fy + fh * 0.3)),
                      255, -1)
    elif bbox is not None:
        bx, by, bw, bh = bbox
        head[:int(by + bh * 0.3)] = 255
    return head


def suit_region(mask, face=None, keep=None):
    """Pixels that become suit: the silhouette without head and ``keep`` (hands, pen)."""
    bbox = player_bbox(mask)
    if bbox is None:
        return np.zeros(mask.shape[:2], np.uint8)
    region = cv2.bitwise_and(mask, cv2.bitwise_not(head_mask(mask.shape, face, bbox)))
    if keep is not None:
        region = cv2.bitwise_and(region, cv2.bitwise_not(keep))
    return region


_SKIN_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def _half(img, interpolation=cv2.INTER_AREA):
    return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=interpolation)


def _full(img, shape, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(img, (shape[1], shape[0]), interpolation=interpolation)


def skin_mask(frame):
    """Skin-coloured pixels (YCrCb range) -- hands stay visible instead of fabric.

    Computed at half resolution (4x fewer pixels, the mask is coarse anyway).
    Clothes in skin tones are kept visible too (known limitation).
    """
    ycrcb = cv2.cvtColor(_half(frame), cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    skin = cv2.morphologyEx(skin, cv2.MORPH_OPEN, _SKIN_KERNEL)
    return _full(cv2.dilate(skin, _SKIN_KERNEL), frame.shape, cv2.INTER_NEAREST)


def paint_fabric(frame, region):
    """Turn ``region`` of ``frame`` into suit fabric (in place).

    Brightness: high in the middle, darker towards the edge of the silhouette
    (distance transform -> arms look round), plus the fine light/dark detail of
    the real clothes (frame minus its blur), so folds stay visible.
    """
    if cv2.countNonZero(region) == 0:
        return frame
    # Only the part of the frame the player occupies is processed (with a margin
    # so the blur sees the surroundings), at half resolution: the brightness map
    # is smooth; the sharp silhouette edge comes from the full-res region.
    x, y, w, h = cv2.boundingRect(region)
    pad = 16
    x0, y0 = max(x - pad, 0) & ~1, max(y - pad, 0) & ~1
    x1, y1 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])
    roi, roi_region = frame[y0:y1, x0:x1], region[y0:y1, x0:x1]

    small_region = _half(roi_region, cv2.INTER_NEAREST)
    gray = _half(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    detail = gray - cv2.GaussianBlur(gray, (0, 0), 6)
    dist = cv2.distanceTransform(small_region, cv2.DIST_L2, 3)
    shade = _SHADE_LUT[np.minimum(dist * 2, 255).astype(np.uint8)]   # dist in full-res px
    value = np.clip(cv2.scaleAdd(detail, 1.2, shade), 0, 255).astype(np.uint8)
    fabric = _full(cv2.merge([value, value, cv2.multiply(value, 0.96)]), roi.shape)
    cv2.copyTo(fabric, roi_region, roi)              # slightly cool white, in place
    contours, _ = cv2.findContours(small_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(roi, [c * 2 for c in contours], -1, (90, 90, 95), 2)
    return frame


# brightness by distance to the silhouette edge: dark rim, bright middle (tube look)
_SHADE_LUT = (150.0 + 90.0 * (1.0 - np.exp(-np.arange(256) / 12.0))).astype(np.float32)


def suit_rect(bbox, face=None):
    """Where the suit goes: (x, y, w, h), or None without a player.

    With a face (x, y, w, h) the suit hangs below the chin, centred on the face
    and scaled to it; otherwise it covers the upper body of the bounding box.
    """
    aspect = SPRITE_H / SPRITE_W
    if face is not None:
        fx, fy, fw, fh = face
        w = round(fw * SUIT_WIDTH_PER_FACE)
        cx = fx + fw / 2
        top = fy + fh * 0.85                 # collar ring right under the chin
    elif bbox is not None:
        bx, by, bw, bh = bbox
        w = round(bw * 0.9)
        cx = bx + bw / 2
        top = by + bh * 0.3                  # below the head
    else:
        return None
    w = max(8, 8 * round(w / 8))             # size in 8 px steps: fewer re-scalings
    return round(cx - w / 2), round(top), w, round(w * aspect)


# --------------------------------------------------------------------------- #
# Overlay
# --------------------------------------------------------------------------- #
def overlay_suit(frame, sprite, rect, mask=None):
    """Alpha-blend ``sprite`` (BGRA) into ``rect`` of ``frame`` (in place).

    Parts outside the frame are clipped; with ``mask`` the suit only covers the
    player's silhouette, never the background next to them.
    """
    if rect is None or sprite is None:
        return frame
    x, y, w, h = rect
    if w <= 0 or h <= 0:
        return frame
    fh, fw = frame.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, fw), min(y + h, fh)
    if x0 >= x1 or y0 >= y1:
        return frame
    part = _scaled(sprite, w, h)[y0 - y:y1 - y, x0 - x:x1 - x]
    alpha = part[:, :, 3]
    if mask is not None:
        alpha = cv2.bitwise_and(alpha, mask[y0:y1, x0:x1])   # masks are 0/255
    # cv2.blendLinear blends in one C++ pass (NumPy needs several temporary arrays)
    weight = alpha.astype(np.float32) * (1.0 / 255.0)
    region = frame[y0:y1, x0:x1]
    frame[y0:y1, x0:x1] = cv2.blendLinear(np.ascontiguousarray(part[:, :, :3]), region,
                                          weight, 1.0 - weight)
    return frame


_SCALE_CACHE = {}


def _scaled(sprite, w, h):
    """Resized sprite, cached per size (the suit size barely changes between frames)."""
    key = (id(sprite), w, h)
    if key not in _SCALE_CACHE:
        if len(_SCALE_CACHE) > 64:
            _SCALE_CACHE.clear()
        _SCALE_CACHE[key] = cv2.resize(sprite, (w, h), interpolation=cv2.INTER_AREA)
    return _SCALE_CACHE[key]


class RectSmoother:
    """Damps frame-to-frame jitter of the suit position (exponential smoothing).

    Only x/y are smoothed; the size already moves in 8 px steps. A jump of more
    than half the suit width (another player, re-entering the frame) is taken
    over immediately instead of sliding across the screen.
    """

    def __init__(self, factor=0.5):
        self.factor = factor
        self._xy = None

    def update(self, rect):
        if rect is None:
            self._xy = None
            return None
        x, y, w, h = rect
        if self._xy is not None:
            px, py = self._xy
            if abs(x - px) <= w / 2 and abs(y - py) <= w / 2:
                x = px + self.factor * (x - px)
                y = py + self.factor * (y - py)
        self._xy = (x, y)
        return round(x), round(y), w, h


def dress(frame, mask, sprite, face=None, smoother=None, keep=None):
    """Put the suit on the player found in ``mask`` (no player -> frame unchanged).

    The fabric follows the silhouette (arms included); ``keep`` (hands, pen)
    stays visible. Collar, panel and patch come from ``sprite`` at the face.
    """
    region = suit_region(mask, face, keep)
    if cv2.countNonZero(region) == 0:
        if smoother is not None:
            smoother.update(None)
        return frame
    paint_fabric(frame, region)
    rect = suit_rect(player_bbox(mask), face)
    if smoother is not None:
        rect = smoother.update(rect)
    return overlay_suit(frame, sprite, rect, mask)


# --------------------------------------------------------------------------- #
# Sprite asset (generated once if missing)
# --------------------------------------------------------------------------- #
def _generate_suit(w=SPRITE_W, h=SPRITE_H):
    """Suit details (BGRA): collar ring, chest panel, mission patch.

    The fabric itself comes from the player's silhouette (``paint_fabric``), so
    everything else in the sprite is transparent.
    """
    ss = 3                                         # supersampling for smooth edges
    W, H = w * ss, h * ss
    color = np.zeros((H, W, 3), np.uint8)
    alpha = np.zeros((H, W), np.uint8)

    def s(v):
        return int(round(v * ss))

    # collar ring around the neck (the neck itself stays transparent)
    cv2.ellipse(alpha, (s(200), s(40)), (s(93), s(42)), 0, 0, 360, 255, -1)
    cv2.ellipse(alpha, (s(200), s(40)), (s(70), s(22)), 0, 0, 360, 0, -1)
    cv2.ellipse(color, (s(200), s(40)), (s(82), s(32)), 0, 0, 360, (190, 190, 200), s(20))
    cv2.ellipse(color, (s(200), s(40)), (s(82), s(32)), 0, 0, 360, (200, 120, 40), s(6))

    # chest control panel with buttons
    cv2.rectangle(alpha, (s(140), s(150)), (s(260), s(230)), 255, -1)
    cv2.rectangle(color, (s(140), s(150)), (s(260), s(230)), (90, 90, 95), -1)
    cv2.rectangle(color, (s(140), s(150)), (s(260), s(230)), (60, 60, 60), s(3))
    for i, c in enumerate([(40, 40, 220), (60, 200, 60), (220, 140, 40)]):
        cv2.circle(color, (s(165 + i * 35), s(175)), s(10), c, -1)
    for i in range(3):
        cv2.line(color, (s(155), s(200 + i * 10)), (s(245), s(200 + i * 10)), (160, 160, 160), s(2))

    # round mission patch on the shoulder (generic, no logo)
    cv2.circle(alpha, (s(300), s(130)), s(26), 255, -1)
    cv2.circle(color, (s(300), s(130)), s(24), (160, 80, 30), -1)
    cv2.circle(color, (s(300), s(130)), s(24), (255, 255, 255), s(3))
    cv2.circle(color, (s(300), s(130)), s(8), (255, 255, 255), -1)

    sprite = np.dstack([color, alpha])
    return cv2.resize(sprite, (w, h), interpolation=cv2.INTER_AREA)


def load_suit(path=SUIT_PATH):
    """Load the suit sprite (BGRA); generate and save it first if it is missing.

    Returns None if it can neither be loaded nor generated (no suit is drawn then).
    """
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, _generate_suit())
            logger.info("Generated space suit sprite -> %s", path)
        except Exception:
            logger.warning("Could not generate space suit sprite", exc_info=True)
            return None
    sprite = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
        logger.warning("Suit sprite at %s missing or without alpha channel", path)
        return None
    return sprite
