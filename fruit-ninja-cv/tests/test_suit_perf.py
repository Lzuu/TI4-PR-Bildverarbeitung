"""PERF-10 (#57): faster suit renderer, same picture.

The reference PNGs were rendered by the float32/full-frame renderer before the
rewrite (commit 2d03a0d), cropped to the area the suit changes.
"""
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
import pytest

from pose_tracking import find_player
from suit_rig import SuitRenderer, _soft_inner_clip, clip_masks
from tools.preview_suit import scene

DATA = Path(__file__).resolve().parent / 'data'
# crop origin (x, y) of each reference image in the 960x720 scene
REFERENCE = {'side': (136, 93), 'cross': (314, 93)}


@pytest.mark.parametrize('kind', sorted(REFERENCE))
def test_render_matches_the_picture_before_the_rewrite(kind):
    expected = cv2.imread(str(DATA / f'suit_ref_{kind}.png'))
    raw, mask, pose, face, keep = scene(kind)
    out = SuitRenderer().render(raw.copy(), raw, mask, pose, face, keep, now=.7)
    x, y = REFERENCE[kind]
    h, w = expected.shape[:2]
    crop = out[y:y+h, x:x+w].astype(int)
    diff = np.abs(crop - expected)
    # The references were rendered on another machine (a Mac). OpenCV rounds a few
    # pixels differently per platform: on Windows even the OLD renderer is 4 grey levels
    # off in places, while old and new renderer agree within 3 on the same machine.
    # Measured on Windows: 20 of 1.25 million pixels are 4 off, none more.
    assert diff.max() <= 4
    assert (diff > 3).mean() < 1e-4
    # outside the crop the old renderer left the camera picture unchanged
    outside = np.ones(mask.shape, bool)
    outside[y:y+h, x:x+w] = False
    assert np.abs(out[outside].astype(int) - raw[outside]).max() <= 3


def _full_frame_clip_masks(player, width):
    """The former full-frame NumPy version, kept as the test oracle."""
    margin = max(2, min(18, round(width*.055)))
    outside_distance = cv2.distanceTransform((player == 0).astype(np.uint8), cv2.DIST_L2, 3)
    exterior = cv2.copyMakeBorder((player == 0).astype(np.uint8), 1, 1, 1, 1,
                                  cv2.BORDER_CONSTANT, value=1)
    cv2.floodFill(exterior, None, (0, 0), 2)
    outer = exterior[1:-1, 1:-1] == 2
    clip = ((player > 0) | (outer & (outside_distance <= margin))).astype(np.uint8)*255
    arm_margin = max(2, min(48, round(width*.16)))
    arm_clip = ((player > 0) | (outer & (outside_distance <= arm_margin))).astype(np.uint8)*255
    return clip, arm_clip


def _selected_player(kind, shift):
    _, mask, _, _, _ = scene(kind, 1., shift)
    mask = mask.copy()
    cv2.circle(mask, (480, 330), 12, 0, -1)          # enclosed hole stays free
    player = find_player(mask)
    selected = np.zeros_like(mask)
    cv2.drawContours(selected, [player.contour], -1, 255, -1)
    return selected & mask, player.box


SCENES = [('side', (0, 0)), ('cross', (0, 0)), ('side', (-420, 0)), ('up', (0, -300))]


# 40: small margins; 200: in between; 340: both margins at their caps (18, 48)
@pytest.mark.parametrize('width', [40., 200., 340.])
@pytest.mark.parametrize('kind,shift', SCENES)
def test_clip_masks_in_the_player_rectangle_equal_the_full_frame_masks(kind, shift, width):
    selected, box = _selected_player(kind, shift)
    clip, arm_clip = clip_masks(selected, box, width)
    expected_clip, expected_arm = _full_frame_clip_masks(selected, width)
    assert np.array_equal(clip, expected_clip)
    assert np.array_equal(arm_clip, expected_arm)


@pytest.mark.parametrize('kind,shift', SCENES)
def test_soft_inner_clip_in_the_player_rectangle_equals_the_full_frame_clip(kind, shift):
    selected, box = _selected_player(kind, shift)
    distance = cv2.distanceTransform(selected, cv2.DIST_L2, 3)
    expected = (np.minimum(distance/1.4, 1)*255).astype(np.uint8)
    assert np.array_equal(_soft_inner_clip(selected, box), expected)


# Opt-in (pytest -m perf): a time budget flakes under load and on shared CI runners.
@pytest.mark.perf
def test_render_of_the_side_scene_stays_within_18_ms():
    raw, mask, pose, face, keep = scene('side')
    renderer = SuitRenderer()
    def run():
        renderer.render(raw.copy(), raw, mask, pose, face, keep, now=0)
    for _ in range(5):
        run()
    times = []
    for _ in range(30):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1000)
    assert statistics.median(times) <= 18


def test_render_with_the_player_found_by_the_loop_gives_the_same_picture():
    raw, mask, pose, face, keep = scene('up')
    alone = SuitRenderer().render(raw.copy(), raw, mask, pose, face, keep, now=0)
    shared = SuitRenderer().render(raw.copy(), raw, mask, pose, face, keep, now=0,
                                   player=find_player(mask))
    assert np.array_equal(alone, shared)
