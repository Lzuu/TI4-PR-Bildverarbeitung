"""EVA suit entry point and inexpensive cloth underlay.

The live articulated renderer lives in suit_rig.py; pose_tracking.py supplies
measured joints. Camera-face pixels remain visible through its clear visor.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FOLD_STRENGTH = 0.3              # how much of the real clothes' folds shows (opaque!)
FOLD_LIMIT = 30                  # strong patterns (stripes, prints) are capped

EDGE_COLOR = (90, 90, 95)        # silhouette outline


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
# Top-level
# --------------------------------------------------------------------------- #
def dress(frame, mask, face=None, keep=None, phase=0.0, *,
          original=None, pose=None, renderer=None, source=None):
    """Render an EVA rig, preserving ``original`` camera pixels for face and pen.

    The game passes one persistent renderer and current measured pose. Older
    callers can omit the keyword arguments and get the limited face/torso path.
    """
    from suit_rig import SuitRenderer
    renderer = SuitRenderer() if renderer is None else renderer
    original = frame.copy() if original is None else original
    return renderer.render(frame, original, mask, pose=pose, face=face, keep=keep,
                           now=phase, source=source)
