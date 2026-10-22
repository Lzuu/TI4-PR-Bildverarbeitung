"""Space suit: dress the cut-out player in a suit (on top of the space background).

Pipeline per frame (only while the space background is active):

  foreground mask (background.py) -> largest blob = player -> bounding box
  + face box (YuNet, if available) -> where the suit goes
  -> alpha-blend the suit sprite there, clipped to the player's silhouette.

The suit is anchored at the face when one is found (collar under the chin,
width relative to the face) -- far more stable than guessing the neck from the
silhouette. Without a face the bounding box is used.

Like the space backdrop, the sprite is generated once into ``assets/suit.png``
(BGRA) if it is missing; replace the file with real artwork to change the look.
"""

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SUIT_PATH = os.path.join(ASSET_DIR, "suit.png")

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


def dress(frame, mask, sprite, face=None, smoother=None):
    """Put the suit on the player found in ``mask`` (no player -> frame unchanged)."""
    rect = suit_rect(player_bbox(mask), face)
    if smoother is not None:
        rect = smoother.update(rect)
    return overlay_suit(frame, sprite, rect, mask)


# --------------------------------------------------------------------------- #
# Sprite asset (generated once if missing)
# --------------------------------------------------------------------------- #
def _generate_suit(w=SPRITE_W, h=SPRITE_H):
    """A white space-suit torso with collar ring, chest panel and mission patch (BGRA)."""
    ss = 3                                         # supersampling for smooth edges
    W, H = w * ss, h * ss
    color = np.zeros((H, W, 3), np.uint8)
    alpha = np.zeros((H, W), np.uint8)

    def s(v):
        return int(round(v * ss))

    # Torso + shoulders: one silhouette (alpha) filled with light grey
    body = np.array([[s(70), s(70)], [s(330), s(70)], [s(345), s(480)], [s(55), s(480)]])
    cv2.fillPoly(alpha, [body], 255)
    for cx in (70, 330):                           # rounded shoulders / upper arms
        cv2.ellipse(alpha, (s(cx), s(120)), (s(70), s(80)), 0, 0, 360, 255, -1)
    cv2.rectangle(alpha, (0, s(150)), (s(40), s(480)), 255, -1)       # arms going down
    cv2.rectangle(alpha, (W - s(40), s(150)), (W, s(480)), 255, -1)
    cv2.ellipse(alpha, (s(200), s(70)), (s(140), s(40)), 0, 180, 360, 255, -1)  # shoulder line

    # vertical shading: white at the top, a bit darker towards the bottom
    shade = np.linspace(245, 200, H, dtype=np.float32)[:, None]
    color[:] = np.repeat(shade, W, axis=1)[:, :, None].astype(np.uint8)
    # arm seams and darker arm sides
    for x in (s(40), W - s(40)):
        cv2.line(color, (x, s(150)), (x, H), (150, 150, 150), s(3))
    for y in (s(300), s(390)):                     # horizontal suit seams (belt line)
        cv2.line(color, (s(60), y), (W - s(60), y), (175, 175, 175), s(2))

    # collar ring around the neck (the neck itself stays transparent)
    cv2.ellipse(color, (s(200), s(40)), (s(85), s(34)), 0, 0, 360, (190, 190, 200), s(16))
    cv2.ellipse(color, (s(200), s(40)), (s(85), s(34)), 0, 0, 360, (200, 120, 40), s(6))
    cv2.ellipse(alpha, (s(200), s(40)), (s(93), s(42)), 0, 0, 360, 255, -1)
    cv2.ellipse(alpha, (s(200), s(40)), (s(70), s(22)), 0, 0, 360, 0, -1)

    # chest control panel with buttons
    cv2.rectangle(color, (s(140), s(150)), (s(260), s(230)), (90, 90, 95), -1)
    cv2.rectangle(color, (s(140), s(150)), (s(260), s(230)), (60, 60, 60), s(3))
    for i, c in enumerate([(40, 40, 220), (60, 200, 60), (220, 140, 40)]):
        cv2.circle(color, (s(165 + i * 35), s(175)), s(10), c, -1)
    for i in range(3):
        cv2.line(color, (s(155), s(200 + i * 10)), (s(245), s(200 + i * 10)), (160, 160, 160), s(2))

    # round mission patch on the shoulder (generic, no logo)
    cv2.circle(color, (s(300), s(130)), s(24), (160, 80, 30), -1)
    cv2.circle(color, (s(300), s(130)), s(24), (255, 255, 255), s(3))
    cv2.circle(color, (s(300), s(130)), s(8), (255, 255, 255), -1)

    # dark outline around the whole silhouette
    contours, _ = cv2.findContours(alpha, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color, contours, -1, (90, 90, 90), s(3))

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
