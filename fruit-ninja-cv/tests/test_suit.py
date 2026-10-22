"""Tests for dressing the cut-out player in a space suit (issue #27). Headless."""

import numpy as np

import suit


def blank(h=400, w=600):
    return np.zeros((h, w), np.uint8)


# --------------------------------------------------------------------------- #
# Player position from the foreground mask
# --------------------------------------------------------------------------- #
def test_no_player_in_empty_mask():
    assert suit.player_bbox(blank()) is None
    assert suit.player_centroid(blank()) is None


def test_bbox_of_a_single_player_blob():
    mask = blank()
    mask[100:300, 200:320] = 255
    assert suit.player_bbox(mask) == (200, 100, 120, 200)


def test_bbox_uses_the_largest_blob():
    mask = blank()
    mask[50:90, 20:60] = 255            # small blob (e.g. a moving chair)
    mask[100:350, 300:450] = 255        # the player
    assert suit.player_bbox(mask) == (300, 100, 150, 250)


def test_tiny_noise_is_not_a_player():
    mask = blank()
    mask[10:14, 10:14] = 255
    assert suit.player_bbox(mask) is None


def test_centroid_of_the_player():
    mask = blank()
    mask[100:300, 200:320] = 255
    cx, cy = suit.player_centroid(mask)
    assert abs(cx - 259.5) < 1 and abs(cy - 199.5) < 1


# --------------------------------------------------------------------------- #
# Where the suit goes
# --------------------------------------------------------------------------- #
def test_suit_hangs_below_the_face_centred_and_scaled():
    face = (270, 80, 60, 70)                       # x, y, w, h
    x, y, w, h = suit.suit_rect((200, 60, 200, 340), face)
    assert abs((x + w / 2) - 300) <= 1             # centred under the face
    assert 80 + 70 * 0.7 <= y <= 80 + 70           # collar just below the chin
    assert w == round(face[2] * suit.SUIT_WIDTH_PER_FACE)


def test_bigger_face_gives_bigger_suit():
    small = suit.suit_rect(None, (100, 100, 40, 45))
    big = suit.suit_rect(None, (100, 100, 80, 90))
    assert big[2] == 2 * small[2]


def test_without_face_the_suit_is_placed_from_the_bbox():
    x, y, w, h = suit.suit_rect((200, 100, 120, 300), None)
    assert abs((x + w / 2) - 260) <= 1
    assert 100 < y < 100 + 300 / 2                 # below the head, upper body


def test_no_rect_without_player():
    assert suit.suit_rect(None, None) is None


def test_smoother_dampens_jitter():
    smooth = suit.RectSmoother(factor=0.5)
    assert smooth.update((100, 100, 200, 200)) == (100, 100, 200, 200)   # first rect as is
    assert smooth.update((110, 100, 200, 200)) == (105, 100, 200, 200)   # halfway there


def test_smoother_resets_when_player_is_gone():
    smooth = suit.RectSmoother(factor=0.5)
    smooth.update((100, 100, 200, 200))
    assert smooth.update(None) is None
    assert smooth.update((300, 300, 200, 200)) == (300, 300, 200, 200)   # no slide from old spot


def test_smoother_follows_big_jumps_immediately():
    smooth = suit.RectSmoother(factor=0.5)
    smooth.update((100, 100, 200, 200))
    assert smooth.update((600, 100, 200, 200)) == (600, 100, 200, 200)   # e.g. other player


# --------------------------------------------------------------------------- #
# Sprite asset
# --------------------------------------------------------------------------- #
def test_sprite_is_generated_once_with_alpha(tmp_path):
    path = tmp_path / "assets" / "suit.png"
    sprite = suit.load_suit(str(path))
    assert path.exists()
    assert sprite.ndim == 3 and sprite.shape[2] == 4
    h, w = sprite.shape[:2]
    assert sprite[0, 0, 3] == 0                     # transparent corner
    assert sprite[h // 2, w // 2, 3] == 255         # opaque chest
    again = suit.load_suit(str(path))               # loaded from file the second time
    assert np.array_equal(again, sprite)


# --------------------------------------------------------------------------- #
# Alpha overlay
# --------------------------------------------------------------------------- #
def _sprite(h=40, w=40):
    s = np.zeros((h, w, 4), np.uint8)
    s[:, :, :3] = 200
    s[10:30, 10:30, 3] = 255                        # opaque square in the middle
    return s


def test_overlay_without_rect_leaves_frame_unchanged():
    frame = np.full((100, 100, 3), 50, np.uint8)
    out = suit.overlay_suit(frame.copy(), _sprite(), None)
    assert np.array_equal(out, frame)


def test_overlay_draws_opaque_and_keeps_transparent_pixels():
    frame = np.full((100, 100, 3), 50, np.uint8)
    out = suit.overlay_suit(frame.copy(), _sprite(), (20, 20, 40, 40))
    assert (out[40, 40] == 200).all()               # opaque part of the sprite
    assert (out[22, 22] == 50).all()                # transparent part
    assert (out[5, 5] == 50).all()                  # outside the rect


def test_overlay_partly_outside_the_frame_does_not_crash():
    frame = np.full((100, 100, 3), 50, np.uint8)
    out = suit.overlay_suit(frame.copy(), _sprite(), (80, 80, 40, 40))
    assert out.shape == frame.shape
    assert (out[95, 95] == 200).all()
    out = suit.overlay_suit(frame.copy(), _sprite(), (200, 200, 40, 40))   # fully outside
    assert np.array_equal(out, frame)


def test_dress_puts_the_suit_on_the_player():
    frame = np.full((400, 600, 3), 50, np.uint8)
    mask = blank()
    mask[100:400, 200:400] = 255
    out = suit.dress(frame.copy(), mask, _sprite(), face=(270, 100, 60, 60))
    assert not np.array_equal(out, frame)
    assert (out[:, :150] == 50).all()               # nothing next to the player


def test_dress_without_player_changes_nothing():
    frame = np.full((400, 600, 3), 50, np.uint8)
    out = suit.dress(frame.copy(), blank(), _sprite(), face=None)
    assert np.array_equal(out, frame)


def test_overlay_only_on_the_player_silhouette():
    frame = np.full((100, 100, 3), 50, np.uint8)
    mask = np.zeros((100, 100), np.uint8)
    mask[:, :40] = 255                              # player only on the left side
    out = suit.overlay_suit(frame.copy(), _sprite(), (20, 20, 40, 40), mask)
    assert (out[35, 35] == 200).all()               # suit on the player
    assert (out[45, 45] == 50).all()                # no suit on the background
