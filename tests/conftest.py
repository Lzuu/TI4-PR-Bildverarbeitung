"""Gemeinsame Hilfen fuer die Tests: synthetische Kamerabilder ohne echte Kamera."""
import numpy as np
import pytest

import fruit_ninja as fn


def background(seed=0, clutter=True):
    """Unruhiger, ueberwiegend grauer Hintergrund in Spielaufloesung (BGR).

    Mit clutter=True kommen ein paar farbige Flaechen dazu (Poster, Moebel ...).
    """
    rng = np.random.default_rng(seed)
    img = rng.integers(60, 200, (fn.H, fn.W, 1), dtype=np.uint8).repeat(3, axis=2)
    img = (img.astype(np.int16) + rng.integers(-10, 10, img.shape)).clip(0, 255).astype(np.uint8)
    if clutter:
        img[40:160, 60:300] = (40, 40, 160)     # rotbraune Flaeche
        img[500:700, 950:1200] = (140, 120, 40)  # blaugraue Flaeche
    return img


def paint(img, center, color, half_size=(60, 15), seed=1):
    """Malt einen stiftfoermigen Gegenstand mit leichtem Helligkeitsrauschen ins Bild."""
    rng = np.random.default_rng(seed)
    out = img.copy()
    x, y = center
    w, h = half_size
    patch = np.full((2 * h, 2 * w, 3), color, np.float32)
    patch *= rng.uniform(0.85, 1.1, (2 * h, 2 * w, 1))
    out[y - h:y + h, x - w:x + w] = patch.clip(0, 255).astype(np.uint8)
    return out


def calibrate(color, bg=None, frames=fn.CALIB_FRAMES):
    """Kalibriert wie im Spiel: Gegenstand fuellt das Kalibrierfeld. Liefert (Modell, Meldung)."""
    bg = background() if bg is None else bg
    obj = bkg = None
    colored = 0.0
    for i in range(frames):
        img = paint(bg, (fn.W // 2, fn.H // 2), color, (fn.CALIB_BOX // 2 + 10,) * 2, seed=i)
        hsv, valid = fn.to_hsv(img)
        o, b, c = fn.calib_histograms(hsv, valid)
        obj = o if obj is None else obj + o
        bkg = b if bkg is None else bkg + b
        colored += c / frames
    return fn.build_model(obj, bkg, colored)


ORANGE = (0, 120, 255)
BLUE = (200, 80, 20)
GREEN = (60, 180, 40)
GRAY = (128, 128, 128)


@pytest.fixture
def bg():
    return background()
