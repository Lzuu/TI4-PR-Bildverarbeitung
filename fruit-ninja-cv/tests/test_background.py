"""Tests for the space-background replacement (foreground segmentation).

Pure, headless logic -- no camera, no GUI. Synthetic numpy frames stand in for
camera images.
"""

import cv2
import numpy as np

from background import foreground_mask, composite_space, load_space

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


def test_composite_does_not_mutate_inputs():
    frame = _solid((0, 0, 255))
    space = _solid((50, 20, 10))
    mask = np.zeros((H, W), np.uint8)
    mask[40:80, 50:100] = 255
    composite_space(frame, mask, space)
    assert tuple(int(c) for c in frame[5, 5]) == (0, 0, 255)   # frame untouched


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
