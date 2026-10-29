"""Tests for the space-background replacement (foreground segmentation).

Pure, headless logic -- no camera, no GUI. Synthetic numpy frames stand in for
camera images.
"""

import cv2
import numpy as np

from background import (foreground_mask, composite_space, load_space,
                        greenscreen_mask, next_mode, MODES, MODE_LABELS,
                        MODE_GREENSCREEN, MODE_NORMAL, MODE_OFF, MODE_YOLO)

H, W = 120, 160


def _solid(bgr):
    return np.full((H, W, 3), bgr, np.uint8)


def test_identical_frames_give_empty_mask():
    ref = _solid((100, 100, 100))
    mask = foreground_mask(ref, ref.copy(), threshold=25)
    assert mask.shape == (H, W)
    assert mask.max() == 0


def test_foreground_block_is_detected():
    ref = _solid((100, 100, 100))
    frame = ref.copy()
    cv2.rectangle(frame, (40, 30), (120, 90), (10, 200, 10), -1)  # the "player"
    mask = foreground_mask(frame, ref, threshold=25)
    assert mask[60, 80] == 255      # inside the block -> foreground
    assert mask[5, 5] == 0          # far corner -> background


def test_foreground_mask_scaled_detects_block_full_size():
    # Segmentation runs at reduced resolution (perf), but the returned mask is
    # upscaled back to the full frame size and still marks the foreground.
    ref = _solid((100, 100, 100))
    frame = ref.copy()
    cv2.rectangle(frame, (40, 30), (120, 90), (10, 200, 10), -1)
    mask = foreground_mask(frame, ref, threshold=25, scale=0.5)
    assert mask.shape == (H, W)      # upscaled back to full frame size
    assert mask[60, 80] == 255       # block still detected
    assert mask[5, 5] == 0           # background still empty


def test_composite_replaces_background_keeps_foreground():
    frame = _solid((0, 0, 255))     # red "player"
    space = _solid((50, 20, 10))    # space backdrop colour
    mask = np.zeros((H, W), np.uint8)
    mask[40:80, 50:100] = 255       # a foreground region
    out = composite_space(frame, mask, space)
    assert out.shape == frame.shape
    assert tuple(int(c) for c in out[60, 70]) == (0, 0, 255)   # mask -> frame kept
    assert tuple(int(c) for c in out[5, 5]) == (50, 20, 10)    # else -> space


def test_composite_writes_in_place():
    # Hot path: composite writes the backdrop into the frame buffer in place
    # rather than allocating a new full-frame array each call.
    frame = _solid((0, 0, 255))
    space = _solid((50, 20, 10))
    mask = np.zeros((H, W), np.uint8)
    mask[40:80, 50:100] = 255
    composite_space(frame, mask, space)
    assert tuple(int(c) for c in frame[5, 5]) == (50, 20, 10)   # background replaced
    assert tuple(int(c) for c in frame[60, 70]) == (0, 0, 255)  # foreground kept


def test_fill_holes_closes_interior_gap():
    ref = _solid((100, 100, 100))
    frame = ref.copy()
    cv2.rectangle(frame, (30, 20), (130, 100), (10, 200, 10), -1)   # player blob
    cv2.rectangle(frame, (70, 50), (90, 70), (100, 100, 100), -1)   # hole == background
    # Without hole filling, the low-contrast "face" region stays background.
    m0 = foreground_mask(frame, ref, threshold=25, scale=1.0, fill_holes=False)
    assert m0[60, 80] == 0
    # With hole filling, the interior gap becomes part of the solid silhouette.
    m1 = foreground_mask(frame, ref, threshold=25, scale=1.0, fill_holes=True)
    assert m1[60, 80] == 255


def test_small_noise_blob_is_dropped():
    ref = _solid((100, 100, 100))
    frame = ref.copy()
    cv2.rectangle(frame, (10, 10), (13, 13), (10, 200, 10), -1)     # tiny speck
    mask = foreground_mask(frame, ref, threshold=25, scale=1.0, fill_holes=True)
    assert mask.max() == 0


def test_load_space_generates_when_missing(tmp_path):
    path = str(tmp_path / "space.png")
    img = load_space((W, H), path=path)
    assert img is not None
    assert img.shape == (H, W, 3)


def test_load_space_resizes_existing(tmp_path):
    path = str(tmp_path / "space.png")
    cv2.imwrite(path, np.full((10, 10, 3), 7, np.uint8))
    img = load_space((W, H), path=path)
    assert img is not None and img.shape == (H, W, 3)


# --------------------------------------------------------------------------- #
# Green-screen chroma key
# --------------------------------------------------------------------------- #
def test_greenscreen_keys_out_green_keeps_player():
    frame = _solid((0, 255, 0))                                 # green screen (BGR)
    cv2.rectangle(frame, (40, 30), (120, 90), (0, 0, 255), -1)  # red "player"
    mask = greenscreen_mask(frame)
    assert mask.shape == (H, W)
    assert mask[60, 80] == 255      # player region kept
    assert mask[5, 5] == 0          # green keyed out -> background


def test_greenscreen_scaled_still_detects_player():
    frame = _solid((0, 255, 0))
    cv2.rectangle(frame, (40, 30), (120, 90), (0, 0, 255), -1)
    mask = greenscreen_mask(frame, scale=0.5)   # segment downscaled, upscale back
    assert mask.shape == (H, W)
    assert mask[60, 80] == 255
    assert mask[5, 5] == 0


def test_greenscreen_keeps_enclosed_green_hole_as_background():
    # Player loops their arms into a ring; the green *inside* the loop must stay
    # keyed out, not filled in. Regression: silhouette filling closed real holes.
    frame = _solid((0, 255, 0))                         # all green
    cv2.circle(frame, (80, 60), 40, (0, 0, 255), 18)    # red ring = the looped arms
    mask = greenscreen_mask(frame)
    assert mask[60, 80] == 0        # centre of the loop = enclosed green -> background
    assert mask[60, 40] == 255      # on the ring = player


def test_greenscreen_full_green_frame_is_all_background():
    # The whole point: a pure green field must be keyed out completely, never
    # left showing through.
    mask = greenscreen_mask(_solid((0, 255, 0)))
    assert mask.max() == 0


def test_greenscreen_keys_out_off_shade_and_shadowed_green():
    # Uneven lighting makes the green vary in brightness/saturation; all of it
    # must still be removed, not just the pixels in a narrow hue band.
    frame = _solid((0, 255, 0))
    cv2.rectangle(frame, (0, 0), (W, 40), (30, 150, 40), -1)      # darker, less sat. green
    cv2.rectangle(frame, (0, 80), (W, H), (60, 90, 30), -1)       # shadowed green
    cv2.rectangle(frame, (50, 45), (110, 75), (0, 0, 255), -1)    # red player in the middle band
    mask = greenscreen_mask(frame)
    assert mask[10, 80] == 0       # off-shade green -> background
    assert mask[110, 80] == 0      # shadowed green -> background
    assert mask[60, 80] == 255     # player kept


# --------------------------------------------------------------------------- #
# Background mode selection
# --------------------------------------------------------------------------- #
def test_modes_are_the_four_options():
    assert set(MODES) == {MODE_GREENSCREEN, MODE_NORMAL, MODE_YOLO, MODE_OFF}
    assert all(m in MODE_LABELS for m in MODES)       # every mode has a menu label


def test_next_mode_cycles_through_all_and_wraps():
    seen, mode = [], MODES[0]
    for _ in range(len(MODES)):
        seen.append(mode)
        mode = next_mode(mode)
    assert seen == list(MODES)       # visits every mode in order
    assert mode == MODES[0]          # wraps back to the first
