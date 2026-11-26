"""Tests for the animated full-body space suit (suit.py).

Pure, headless logic -- no camera, no GUI. Synthetic masks/frames stand in for
the player. The suit now dresses the *whole* silhouette (head -> helmet, hands
-> gloves); only the pen (``keep``) stays visible.
"""

import cv2
import numpy as np

import suit


H, W = 400, 600


def _blank():
    return np.zeros((H, W), np.uint8)


def _body():
    """A silhouette with head, torso and a raised arm ending in a hand."""
    m = _blank()
    cv2.circle(m, (300, 80), 45, 255, -1)              # head
    cv2.rectangle(m, (240, 120), (360, 330), 255, -1)  # torso
    cv2.rectangle(m, (200, 140), (240, 270), 255, -1)  # raised arm
    cv2.circle(m, (220, 130), 22, 255, -1)             # hand (becomes a glove)
    return m


# --------------------------------------------------------------------------- #
# Fabric
# --------------------------------------------------------------------------- #
def test_fabric_is_light_and_opaque():
    frame = np.full((H, W, 3), 40, np.uint8)           # dark shirt
    region = _blank()
    cv2.rectangle(region, (200, 150), (360, 290), 255, -1)
    suit.paint_fabric(frame, region)
    assert int(frame[220, 280].mean()) > 170           # fabric is bright
    assert int(frame[5, 5].mean()) == 40               # background untouched


def test_fabric_is_darker_at_the_edge_than_in_the_middle():
    frame = np.full((H, W, 3), 100, np.uint8)
    region = _blank()
    cv2.rectangle(region, (200, 150), (360, 290), 255, -1)
    suit.paint_fabric(frame, region)
    middle = int(frame[220, 280, 0])
    near_edge = int(frame[220, 208, 0])
    assert middle > near_edge + 15                     # rounded, tube-like shading


# --------------------------------------------------------------------------- #
# dress (top level)
# --------------------------------------------------------------------------- #
def test_dress_puts_fabric_on_the_body_and_a_helmet_on_the_head():
    frame = np.full((H, W, 3), 50, np.uint8)
    mask = _body()
    face = (255, 35, 90, 90)
    suit.dress(frame, mask, face=face, phase=0.3)
    assert int(frame[220, 300].mean()) > 150           # torso is now fabric
    assert abs(int(frame[70, 300, 0]) - 50) <= 10     # camera face survives the visor
    assert int(frame[380, 20].mean()) == 50            # far background untouched


# -- CR-05 (#41): the protected pen survives every render stage of dress -----
def _pen_scene():
    frame = np.full((240, 320, 3), 60, np.uint8)
    mask = np.zeros((240, 320), np.uint8)
    mask[10:240, 35:285] = 255                         # person
    return frame, mask, (120, 20, 50, 60)              # face box x, y, w, h


def _pen_at(frame, center):
    keep = np.zeros(frame.shape[:2], np.uint8)
    cv2.circle(keep, center, 9, 255, -1)
    frame[keep > 0] = (180, 0, 255)                    # pink pen pixels
    return keep


def test_pen_in_front_of_the_helmet_stays_visible():
    frame, mask, face = _pen_scene()
    keep = _pen_at(frame, (145, 50))                   # over the visor
    before = frame.copy()
    out = suit.dress(frame, mask, face, keep, phase=1.0)
    assert (out[keep > 0] == before[keep > 0]).all()


def test_suit_is_still_drawn_around_the_protected_pen():
    frame, mask, face = _pen_scene()
    keep = _pen_at(frame, (145, 115))
    before = frame.copy()
    out = suit.dress(frame, mask, face, keep, phase=0.0)
    assert not (out[200, 60] == before[200, 60]).all()   # fabric elsewhere
    assert not (out[8, 145] == before[8, 145]).all()     # white shell above the face
