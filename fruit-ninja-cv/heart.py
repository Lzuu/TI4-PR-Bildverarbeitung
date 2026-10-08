"""A polished heart icon for the lives display.

The sprite is generated once (and cached per size) from the classic parametric
heart curve, filled with a vertical red gradient, a glossy highlight and a dark
outline, anti-aliased via supersampling. It is stored as a BGRA image and
alpha-blended onto the HUD -- so it looks like a real game asset without needing
an external image file.
"""

import numpy as np
import cv2

_CACHE = {}


def _build_heart(size):
    ss = size * 4                       # supersample for anti-aliasing
    t = np.linspace(0, 2 * np.pi, 600)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())
    pad = int(ss * 0.08)
    px = (pad + x * (ss - 2 * pad)).astype(np.int32)
    py = (pad + (1 - y) * (ss - 2 * pad)).astype(np.int32)   # flip: lobes on top
    pts = np.stack([px, py], axis=1)

    mask = np.zeros((ss, ss), np.uint8)
    cv2.fillPoly(mask, [pts], 255, cv2.LINE_AA)

    # Vertical gradient (BGR): lighter red at the top, deeper red at the bottom
    top = np.array([95, 95, 255], np.float32)
    bottom = np.array([35, 30, 185], np.float32)
    ramp = np.linspace(0, 1, ss, dtype=np.float32)[:, None]
    grad = (top * (1 - ramp) + bottom * ramp).astype(np.uint8)
    bgr = np.repeat(grad[:, None, :], ss, axis=1)

    # Glossy highlight (upper-left), blended softly
    hl = np.zeros((ss, ss), np.uint8)
    cv2.ellipse(hl, (int(ss * 0.37), int(ss * 0.34)),
                (int(ss * 0.13), int(ss * 0.095)), -30, 0, 360, 255, -1, cv2.LINE_AA)
    hl = cv2.GaussianBlur(hl, (0, 0), ss * 0.02)
    hlf = (hl.astype(np.float32) / 255.0)[:, :, None] * 0.7
    bgr = (bgr.astype(np.float32) * (1 - hlf) + 255.0 * hlf).astype(np.uint8)

    # Dark outline along the curve
    cv2.polylines(bgr, [pts], True, (25, 20, 90), max(2, ss // 90), cv2.LINE_AA)

    bgra = np.dstack([bgr, mask])
    return cv2.resize(bgra, (size, size), interpolation=cv2.INTER_AREA)


def get_heart_sprite(size):
    size = max(6, int(size))
    if size not in _CACHE:
        _CACHE[size] = _build_heart(size)
    return _CACHE[size]


def draw_heart(frame, center, size):
    """Alpha-blend a heart sprite centred at ``center`` (x, y)."""
    sprite = get_heart_sprite(size)
    sh, sw = sprite.shape[:2]
    x0 = int(center[0] - sw / 2)
    y0 = int(center[1] - sh / 2)
    x1, y1 = x0 + sw, y0 + sh

    fh, fw = frame.shape[:2]
    # Clip to frame bounds
    fx0, fy0 = max(0, x0), max(0, y0)
    fx1, fy1 = min(fw, x1), min(fh, y1)
    if fx0 >= fx1 or fy0 >= fy1:
        return
    sx0, sy0 = fx0 - x0, fy0 - y0
    sx1, sy1 = sx0 + (fx1 - fx0), sy0 + (fy1 - fy0)

    region = sprite[sy0:sy1, sx0:sx1]
    alpha = region[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[fy0:fy1, fx0:fx1]
    frame[fy0:fy1, fx0:fx1] = (region[:, :, :3].astype(np.float32) * alpha
                               + roi.astype(np.float32) * (1 - alpha)).astype(np.uint8)
