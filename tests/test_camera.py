"""Teilaufgabe 1: Bilderfassung. Braucht die echte Kamera, laeuft daher nicht in der CI."""
import time

import pytest

from camera import CAMERA_INDEX, open_camera

pytestmark = pytest.mark.camera


@pytest.fixture(scope="module")
def cap():
    try:
        cap = open_camera(CAMERA_INDEX)
    except RuntimeError:
        pytest.skip(f"Keine Kamera an Index {CAMERA_INDEX}")
    yield cap
    cap.release()


def test_kamera_liefert_full_hd(cap):
    ok, frame = cap.read()
    assert ok
    assert frame.shape == (1080, 1920, 3)


def test_kamera_schafft_mindestens_25_fps(cap):
    for _ in range(10):       # Anlaufzeit der Kamera abwarten
        cap.read()
    start, frames = time.time(), 0
    while time.time() - start < 2:
        frames += cap.read()[0]
    assert frames / 2 >= 25
