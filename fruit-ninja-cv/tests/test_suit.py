"""Tests for the animated body space suit (suit.py).

Pure, headless logic -- no camera, no GUI. Synthetic masks/frames stand in for
the player. The suit dresses the whole silhouette (head and hands -> gloves) in
tinted fabric; there is no helmet and only the pen (``keep``) stays visible.
"""

import cv2
import numpy as np

import config
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
# Player position
# --------------------------------------------------------------------------- #
def test_no_player_in_empty_mask():
    assert suit.player_bbox(_blank()) is None
    assert suit.player_centroid(_blank()) is None


def test_bbox_uses_the_largest_blob():
    m = _body()
    cv2.circle(m, (20, 20), 4, 255, -1)                # tiny noise blob
    x, y, w, h = suit.player_bbox(m)
    assert x <= 200 and y <= 80 and x + w >= 360       # covers the whole body
    assert (x, y) != (16, 16)                          # not the noise


# --------------------------------------------------------------------------- #
# Full-body region (whole silhouette incl. head becomes suit; only the pen free)
# --------------------------------------------------------------------------- #
def test_region_covers_head_and_hands():
    region = suit.full_body_region(_body())
    assert region[80, 300] == 255        # head is part of the suit
    assert region[130, 220] == 255       # hand is covered (glove)


def test_region_excludes_the_pen():
    keep = _blank()
    cv2.rectangle(keep, (290, 200), (310, 240), 255, -1)   # the pen marker
    region = suit.full_body_region(_body(), keep=keep)
    assert region[220, 300] == 0         # pen stays visible, not fabric
    assert region[80, 300] == 255        # head still covered


def test_region_empty_without_player():
    assert suit.full_body_region(_blank()).max() == 0


# --------------------------------------------------------------------------- #
# Fabric
# --------------------------------------------------------------------------- #
def test_fabric_is_tinted_not_white_and_opaque():
    frame = np.full((H, W, 3), 40, np.uint8)           # dark shirt
    region = _blank()
    cv2.rectangle(region, (200, 150), (360, 290), 255, -1)
    suit.paint_fabric(frame, region)
    b, g, r = (int(c) for c in frame[220, 280])
    assert r > b + 20                                  # orange tint (R dominant), not plain white/grey
    assert (b + g + r) / 3 > 90                         # fabric is bright and opaque
    assert int(frame[5, 5].mean()) == 40               # background untouched


def test_fabric_is_darker_at_the_edge_than_in_the_middle():
    frame = np.full((H, W, 3), 100, np.uint8)
    region = _blank()
    cv2.rectangle(region, (200, 150), (360, 290), 255, -1)
    suit.paint_fabric(frame, region)
    middle = int(frame[220, 280, 2])                   # R channel (dominant in the orange tint)
    near_edge = int(frame[220, 208, 2])
    assert middle > near_edge + 15                     # rounded, tube-like shading


# --------------------------------------------------------------------------- #
# Chest panel placement + animation
# --------------------------------------------------------------------------- #
def test_chest_rect_scales_with_the_face():
    small = suit.chest_rect(bbox=None, face=(280, 40, 40, 40))
    big = suit.chest_rect(bbox=None, face=(280, 40, 80, 80))
    assert big[2] > small[2]
    assert suit.chest_rect(bbox=None, face=None) is None


def test_chest_panel_leds_blink_over_time():
    rect = (250, 120, 120, 150)
    a = np.full((H, W, 3), 180, np.uint8)
    b = a.copy()
    suit.draw_chest_panel(a, rect, phase=0.0)
    suit.draw_chest_panel(b, rect, phase=0.9)
    assert not np.array_equal(a, np.full((H, W, 3), 180, np.uint8))   # something was drawn
    assert not np.array_equal(a[135:210, 290:350], b[135:210, 290:350])  # LEDs animated


# --------------------------------------------------------------------------- #
# RectSmoother
# --------------------------------------------------------------------------- #
def test_smoother_damps_small_moves():
    s = suit.RectSmoother(factor=0.5)
    s.update((100, 100, 40, 60))
    x, y, w, h = s.update((120, 100, 40, 60))
    assert 100 < x < 120                               # eased towards the new x


def test_smoother_jumps_for_a_big_move():
    s = suit.RectSmoother(factor=0.5)
    s.update((100, 100, 40, 60))
    x, _, _, _ = s.update((300, 100, 40, 60))          # > half width away
    assert x == 300                                    # taken over immediately


def test_smoother_resets_when_player_gone():
    s = suit.RectSmoother()
    s.update((100, 100, 40, 60))
    assert s.update(None) is None
    assert s._xy is None


# --------------------------------------------------------------------------- #
# dress (top level)
# --------------------------------------------------------------------------- #
def test_dress_puts_fabric_on_the_whole_body_incl_head():
    frame = np.full((H, W, 3), 50, np.uint8)
    mask = _body()
    face = (255, 35, 90, 90)
    suit.dress(frame, mask, face=face, phase=0.3)
    assert int(frame[220, 300].mean()) != 50           # torso is now fabric
    assert int(frame[80, 300].mean()) != 50            # head is now fabric too (no helmet)
    assert int(frame[380, 20].mean()) == 50            # far background untouched


def test_dress_without_player_changes_nothing_and_resets():
    frame = np.full((H, W, 3), 50, np.uint8)
    smoother = suit.RectSmoother()
    smoother.update((100, 100, 40, 60))
    out = suit.dress(frame, _blank(), smoother=smoother)
    assert np.array_equal(out, np.full((H, W, 3), 50, np.uint8))
    assert smoother._xy is None
