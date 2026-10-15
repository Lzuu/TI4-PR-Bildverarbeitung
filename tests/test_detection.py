"""Teilaufgabe 2: Stifterkennung (Kalibrierung + Histogramm-Rueckprojektion)."""
import math

import numpy as np
import pytest

import fruit_ninja as fn
from conftest import BLUE, GRAY, GREEN, ORANGE, background, calibrate, paint


@pytest.mark.parametrize("color", [ORANGE, BLUE, GREEN], ids=["orange", "blau", "gruen"])
def test_kalibrierter_gegenstand_wird_ueberall_gefunden(color, bg):
    model, msg = calibrate(color, bg)
    assert model is not None, msg
    rng = np.random.default_rng(42)
    for _ in range(20):
        target = (int(rng.integers(200, 1080)), int(rng.integers(220, 480)))
        pen, _ = fn.find_pen(*fn.to_hsv(paint(bg, target, color)), model)
        assert pen is not None, f"nicht gefunden bei {target}"
        assert math.dist(pen, target) < 5


@pytest.mark.parametrize("color", [ORANGE, BLUE, GREEN], ids=["orange", "blau", "gruen"])
def test_kein_fehlalarm_ohne_gegenstand(color, bg):
    model, _ = calibrate(color, bg)
    pen, mask = fn.find_pen(*fn.to_hsv(bg), model)
    assert pen is None


def test_andere_farbe_wird_ignoriert(bg):
    model, _ = calibrate(ORANGE, bg)
    pen, _ = fn.find_pen(*fn.to_hsv(paint(bg, (400, 300), GREEN)), model)
    assert pen is None


def test_grauer_gegenstand_wird_abgelehnt(bg):
    model, msg = calibrate(GRAY, bg)
    assert model is None
    assert "Farbe" in msg


def test_warnung_wenn_farbe_auch_im_hintergrund_ist():
    bg = background()
    bg[:, :] = (0, 120, 255)                   # Hintergrund komplett in Stiftfarbe
    bg = paint(bg, (200, 200), GRAY, (100, 100))
    model, msg = calibrate(ORANGE, bg)
    assert model is not None
    assert "Hintergrund" in msg


def test_erkennung_bei_dunklerem_licht(bg):
    """Nach der Kalibrierung wird der Gegenstand auch bei 30 % weniger Licht gefunden."""
    model, _ = calibrate(ORANGE, bg)
    darker = (paint(bg, (400, 300), ORANGE).astype(np.float32) * 0.7).astype(np.uint8)
    pen, _ = fn.find_pen(*fn.to_hsv(darker), model)
    assert pen is not None
    assert math.dist(pen, (400, 300)) < 5


def test_zu_kleiner_fleck_ist_kein_stift(bg):
    model, _ = calibrate(ORANGE, bg)
    tiny = paint(bg, (400, 300), ORANGE, half_size=(3, 3))
    pen, _ = fn.find_pen(*fn.to_hsv(tiny), model)
    assert pen is None
