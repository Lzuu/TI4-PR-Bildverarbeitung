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
# Suit fabric on the silhouette: moves with the arms
# --------------------------------------------------------------------------- #
FACE = (270, 60, 60, 70)                            # x, y, w, h


def _body_with_raised_arm():
    """Silhouette: head, torso and the left arm raised up and to the side."""
    mask = blank()
    mask[40:140, 260:340] = 255                     # head
    mask[140:400, 220:380] = 255                    # torso
    mask[60:200, 120:180] = 255                     # raised upper arm, beside the head
    mask[180:220, 120:240] = 255                    # shoulder joint
    return mask


def test_suit_region_covers_the_raised_arm():
    region = suit.suit_region(_body_with_raised_arm(), FACE)
    assert region[100, 150] == 255                  # arm above shoulder height
    assert region[300, 300] == 255                  # torso


def test_suit_region_leaves_the_head_free():
    region = suit.suit_region(_body_with_raised_arm(), FACE)
    assert region[90, 300] == 0                     # face
    assert region[45, 300] == 0                     # hair above the face


def test_suit_region_keeps_hands_and_pen_visible():
    mask = _body_with_raised_arm()
    keep = blank()
    keep[60:80, 120:180] = 255                      # hand (skin) or pen at the arm's end
    region = suit.suit_region(mask, FACE, keep)
    assert region[70, 150] == 0
    assert region[120, 150] == 255


def test_suit_region_without_face_starts_below_the_head():
    mask = blank()
    mask[50:400, 200:400] = 255
    region = suit.suit_region(mask, None)
    assert region[60, 300] == 0                     # top of the silhouette = head
    assert region[350, 300] == 255


def test_skin_is_detected_but_suit_fabric_is_not():
    frame = np.zeros((100, 200, 3), np.uint8)
    frame[:, :100] = (120, 150, 200)                # BGR skin tone
    frame[:, 100:] = (200, 60, 40)                  # blue jumper
    skin = suit.skin_mask(frame)
    assert skin[50, 50] == 255
    assert skin[50, 150] == 0


def test_fabric_is_light_and_keeps_the_folds():
    frame = np.full((200, 200, 3), (40, 80, 160), np.uint8)    # dark red shirt
    frame[:, 95:105] = (20, 40, 80)                            # a fold (darker stripe)
    region = np.zeros((200, 200), np.uint8)
    region[20:180, 20:180] = 255
    out = suit.paint_fabric(frame.copy(), region)
    assert out[100, 60].mean() > 150                           # suit is light
    assert abs(int(out[100, 60, 0]) - int(out[100, 60, 2])) < 30   # and nearly grey/white
    assert out[100, 100].mean() < out[100, 60].mean()          # fold still visible
    assert (out[5, 5] == frame[5, 5]).all()                    # outside untouched


def test_fabric_is_darker_at_the_edge_than_in_the_middle():
    frame = np.full((200, 200, 3), 120, np.uint8)
    region = np.zeros((200, 200), np.uint8)
    region[20:180, 20:180] = 255
    out = suit.paint_fabric(frame.copy(), region)
    assert out[100, 22].mean() < out[100, 100].mean()          # rounded, tube-like look


def test_dress_moves_with_the_arm():
    """Same body, arm down vs. arm up: the suit follows the arm."""
    frame = np.full((400, 600, 3), 60, np.uint8)
    up = _body_with_raised_arm()
    down = blank()
    down[40:140, 260:340] = 255
    down[140:400, 220:380] = 255
    down[180:380, 160:220] = 255                    # arm hanging down
    out_up = suit.dress(frame.copy(), up, suit.load_suit(), FACE)
    out_down = suit.dress(frame.copy(), down, suit.load_suit(), FACE)
    assert out_up[100, 150].mean() > 120 and out_down[100, 150].mean() == 60
    assert out_down[300, 190].mean() > 120 and out_up[300, 190].mean() == 60


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
    assert sprite[190, 200, 3] == 255               # opaque chest panel
    # the suit fabric itself comes from the silhouette, so the sprite only holds
    # the details (collar, panel, patch) -- the lower torso is transparent
    assert sprite[int(h * 0.85), w // 2, 3] == 0
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
