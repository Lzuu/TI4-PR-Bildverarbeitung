"""EVA suit compatibility helpers and inexpensive cloth underlay.

The live articulated renderer lives in suit_rig.py; pose_tracking.py supplies
measured joints. Camera-face pixels remain visible through its clear visor.
The pure placement/cloth helpers remain available for tests and limited fallback.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

MIN_PLAYER_FRAC = 0.01           # blobs smaller than this frame fraction are noise
SUIT_WIDTH_PER_FACE = 3.2        # chest-panel reference width = face width * this
SPRITE_H, SPRITE_W = 480, 400    # aspect used when placing the chest panel from a bbox
FOLD_STRENGTH = 0.3              # how much of the real clothes' folds shows (opaque!)
FOLD_LIMIT = 30                  # strong patterns (stripes, prints) are capped

EDGE_COLOR = (90, 90, 95)        # silhouette outline
FABRIC_COOL = 0.96               # retained legacy palette constant


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


def full_body_region(mask, keep=None):
    """Pixels that become suit: the whole silhouette, minus ``keep`` (the pen)."""
    if player_bbox(mask) is None:
        return np.zeros(mask.shape[:2], np.uint8)
    region = mask.copy()
    if keep is not None:
        region = cv2.bitwise_and(region, cv2.bitwise_not(keep))
    return region


def head_box(face=None, bbox=None):
    """Head region (x, y, w, h) for the helmet: the face box, or the top of the bbox."""
    if face is not None:
        return tuple(int(v) for v in face)
    if bbox is not None:
        bx, by, bw, bh = bbox
        hw, hh = int(bw * 0.5), int(bh * 0.28)
        return int(bx + bw / 2 - hw / 2), int(by), hw, hh
    return None


# --------------------------------------------------------------------------- #
# Fabric (silhouette -> suit cloth)
# --------------------------------------------------------------------------- #
def _half(img, interpolation=cv2.INTER_AREA):
    return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=interpolation)


def _full(img, shape, interpolation=cv2.INTER_LINEAR):
    return cv2.resize(img, (shape[1], shape[0]), interpolation=interpolation)


# brightness by distance to the silhouette edge: dark rim, bright middle (tube look)
_SHADE_LUT = (150.0 + 90.0 * (1.0 - np.exp(-np.arange(256) / 12.0))).astype(np.float32)


def paint_fabric(frame, region):
    """Turn ``region`` of ``frame`` into cool-white suit fabric (in place).

    Brightness is high in the middle and darker towards the silhouette edge
    (distance transform -> rounded limbs), plus a little of the fine light/dark
    detail of the real clothes (capped, so the fabric stays opaque). The
    silhouette gets a thin dark outline so the suit shape reads clearly.
    """
    if cv2.countNonZero(region) == 0:
        return frame
    # Process only the player's bounding area (plus a margin so the blur sees the
    # surroundings), at half resolution: the brightness map is smooth; the sharp
    # edge comes from the full-res region mask.
    x, y, w, h = cv2.boundingRect(region)
    pad = 16
    x0, y0 = max(x - pad, 0) & ~1, max(y - pad, 0) & ~1
    x1, y1 = min(x + w + pad, frame.shape[1]), min(y + h + pad, frame.shape[0])
    roi, roi_region = frame[y0:y1, x0:x1], region[y0:y1, x0:x1]

    small_region = _half(roi_region, cv2.INTER_NEAREST)
    gray = _half(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)).astype(np.float32)
    detail = np.clip(gray - cv2.GaussianBlur(gray, (0, 0), 6), -FOLD_LIMIT, FOLD_LIMIT)
    dist = cv2.distanceTransform(small_region, cv2.DIST_L2, 3)
    shade = _SHADE_LUT[(np.minimum(dist, 127.5) * 2).astype(np.uint8)]
    value = np.clip(cv2.scaleAdd(detail, FOLD_STRENGTH, shade), 0, 255)
    v = value.astype(np.uint8)
    small_fabric = cv2.merge([cv2.multiply(v, .94), cv2.multiply(v, .97), v])    # slightly cool white
    fabric = _full(small_fabric, roi.shape)
    cv2.copyTo(fabric, roi_region, roi)
    contours, _ = cv2.findContours(small_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(roi, [c * 2 for c in contours], -1, EDGE_COLOR, 2)
    return frame


# --------------------------------------------------------------------------- #
# Animated helmet
# --------------------------------------------------------------------------- #


def draw_helmet(frame, head, phase=0.0, original=None):
    """Compatibility helper: local EVA shell with a near-clear camera-face visor."""
    if head is not None:
        from suit_rig import SuitRenderer
        SuitRenderer()._helmet(frame, frame.copy() if original is None else original,
                               (head, 0.), phase)
    return frame


# --------------------------------------------------------------------------- #
# Animated chest panel
# --------------------------------------------------------------------------- #
def chest_rect(bbox, face=None):
    """Reference box (x, y, w, h) used to place the chest panel, or None."""
    aspect = SPRITE_H / SPRITE_W
    if face is not None:
        fx, fy, fw, fh = face
        w = round(fw * SUIT_WIDTH_PER_FACE)
        cx = fx + fw / 2
        top = fy + fh * 0.85
    elif bbox is not None:
        bx, by, bw, bh = bbox
        w = round(bw * 0.9)
        cx = bx + bw / 2
        top = by + bh * 0.3
    else:
        return None
    w = max(8, 8 * round(w / 8))
    return round(cx - w / 2), round(top), w, round(w * aspect)


_LED_COLORS = [(40, 40, 220), (60, 200, 60), (220, 150, 40)]


def draw_chest_panel(frame, rect, phase=0.0):
    """Draw an animated control panel (blinking LEDs) on the chest, in place."""
    if rect is None:
        return frame
    x, y, w, h = rect
    pw = max(8, int(w * 0.34))
    ph = max(6, int(pw * 0.62))
    px, py = int(x + w / 2 - pw / 2), int(y + h * 0.13)

    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (72, 72, 80), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (px, py), (px + pw, py + ph), (45, 45, 50), max(1, pw // 28), cv2.LINE_AA)
    for i, base in enumerate(_LED_COLORS):
        blink = 0.35 + 0.65 * (0.5 + 0.5 * np.sin(phase * 3.0 + i * 2.1))
        col = tuple(int(c * blink) for c in base)
        lx = px + int(pw * (0.24 + 0.26 * i))
        cv2.circle(frame, (lx, py + int(ph * 0.33)), max(2, pw // 13), col, -1, cv2.LINE_AA)
    for i in range(3):
        ly = py + int(ph * (0.6 + 0.12 * i))
        cv2.line(frame, (px + int(pw * 0.15), ly), (px + int(pw * 0.85), ly),
                 (155, 155, 165), max(1, pw // 36), cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------- #
# Position smoothing + top-level
# --------------------------------------------------------------------------- #
class RectSmoother:
    """Damps frame-to-frame jitter of the chest-panel position (exponential).

    A jump of more than half the width (another player entering the frame) is
    taken over immediately instead of sliding across the screen.
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


def dress(frame, mask, face=None, keep=None, phase=0.0, smoother=None, *,
          original=None, pose=None, renderer=None, source=None):
    """Render an EVA rig, preserving ``original`` camera pixels for face and pen.

    The game passes one persistent renderer and current measured pose. Older
    callers can omit the keyword arguments and get the limited face/torso path.
    """
    from suit_rig import SuitRenderer
    if player_bbox(mask) is None and smoother is not None:
        smoother.update(None)
    renderer = SuitRenderer() if renderer is None else renderer
    original = frame.copy() if original is None else original
    return renderer.render(frame, original, mask, pose=pose, face=face, keep=keep,
                           now=phase, source=source)
