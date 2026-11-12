"""Tests for the YOLOv8-seg body segmentation decode and segmenter.

Pure, headless logic -- no ONNX model, no camera, no GPU. Synthetic raw-output
tensors stand in for the network; a FakeNet drives the segmenter.
"""

import cv2
import numpy as np
import pytest

import config
import yolo_seg as y


# --------------------------------------------------------------------------- #
# Raw-output builders (YOLOv8-seg layout: output0 (1, 4+80+32, A))
# --------------------------------------------------------------------------- #
CH = 4 + y.NUM_CLASSES + 32          # 116


def _output0(anchors):
    """anchors = list of (cx, cy, w, h, class_idx, score, coeffs(32,))."""
    out = np.zeros((1, CH, len(anchors)), np.float32)
    for a, (cx, cy, w, h, cls, score, coeffs) in enumerate(anchors):
        out[0, 0:4, a] = (cx, cy, w, h)
        out[0, 4 + cls, a] = score
        out[0, 4 + y.NUM_CLASSES:, a] = coeffs
    return out


# --------------------------------------------------------------------------- #
# person_detections
# --------------------------------------------------------------------------- #
def test_person_detections_keeps_only_confident_persons():
    out0 = _output0([
        (50, 60, 20, 40, y.PERSON_CLASS, 0.9, np.ones(32)),   # person, kept
        (10, 10, 10, 10, y.PERSON_CLASS, 0.1, np.zeros(32)),  # person, too low
        (80, 80, 30, 30, 16,             0.9, np.zeros(32)),  # a dog, not a person
    ])
    xywh, confs, coeffs = y.person_detections(out0, conf=0.4)
    assert len(xywh) == 1
    assert np.allclose(xywh[0], [40, 40, 20, 40])    # cx,cy,w,h -> top-left x,y,w,h
    assert confs[0] == pytest.approx(0.9)
    assert coeffs.shape == (1, 32)


def test_person_detections_empty_when_nothing_passes():
    out0 = _output0([(50, 60, 20, 40, y.PERSON_CLASS, 0.2, np.zeros(32))])
    xywh, confs, coeffs = y.person_detections(out0, conf=0.4)
    assert len(xywh) == 0


# --------------------------------------------------------------------------- #
# nms
# --------------------------------------------------------------------------- #
def test_nms_collapses_overlapping_boxes():
    xywh = np.array([[10, 10, 40, 40], [12, 12, 40, 40]], np.float32)   # near-identical
    confs = np.array([0.9, 0.8], np.float32)
    assert len(y.nms(xywh, confs, conf=0.4)) == 1


def test_nms_keeps_separate_boxes():
    xywh = np.array([[0, 0, 20, 20], [100, 100, 20, 20]], np.float32)
    confs = np.array([0.9, 0.9], np.float32)
    assert len(y.nms(xywh, confs, conf=0.4)) == 2


def test_nms_empty_input():
    assert y.nms(np.empty((0, 4)), np.empty((0,)), conf=0.4) == []


# --------------------------------------------------------------------------- #
# person_mask / assemble_mask
# --------------------------------------------------------------------------- #
def test_person_mask_is_inside_the_box_only():
    protos = np.zeros((32, 8, 8), np.float32)
    protos[0] = 50.0                                 # big positive -> sigmoid ~1
    coeffs = np.zeros(32, np.float32)
    coeffs[0] = 1.0
    m = y.person_mask(coeffs, protos, box=(4, 4, 8, 8), size=16, thresh=0.5)
    assert m.shape == (16, 16)
    assert m[8, 8] == 255        # inside the box, mask active
    assert m[0, 0] == 0          # outside the box -> cropped away


def test_assemble_mask_unions_detections():
    protos = np.zeros((32, 8, 8), np.float32)
    protos[0] = 50.0
    coeffs = np.array([[1.0] + [0.0] * 31, [1.0] + [0.0] * 31], np.float32)
    xywh = np.array([[0, 0, 4, 4], [12, 12, 4, 4]], np.float32)
    full = y.assemble_mask(xywh, coeffs, idx=[0, 1], protos=protos, size=16, thresh=0.5)
    assert full[1, 1] == 255 and full[13, 13] == 255     # both boxes present
    assert full[8, 8] == 0                               # gap between them stays empty


def test_assemble_mask_no_detections_is_blank():
    protos = np.zeros((32, 8, 8), np.float32)
    full = y.assemble_mask(np.empty((0, 4)), np.empty((0, 32)), [], protos, 16, 0.5)
    assert full.max() == 0


# --------------------------------------------------------------------------- #
# Continuous mask (finer edges: threshold at full resolution, not the input grid)
# --------------------------------------------------------------------------- #
def test_assemble_prob_is_continuous_with_intermediate_edge_values():
    protos = np.zeros((32, 4, 4), np.float32)
    protos[0, :, 2:] = 6.0                            # right half ~1, left half ~0 (a ramp)
    protos[0, :, :2] = -6.0
    coeffs = np.array([[1.0] + [0.0] * 31], np.float32)
    xywh = np.array([[0, 0, 8, 8]], np.float32)
    prob = y.assemble_prob(xywh, coeffs, [0], protos, size=8)
    assert prob.dtype == np.float32
    assert prob.max() > 0.9 and prob.min() < 0.1
    assert ((prob > 0.2) & (prob < 0.8)).any()       # genuine intermediate values at the edge


def test_full_res_threshold_gives_a_finer_edge_than_nearest():
    """Upscaling the continuous mask (bilinear) and thresholding at full
    resolution follows a diagonal boundary at far more sub-grid positions than
    nearest-upscaling a mask already thresholded on the coarse input grid."""
    size, big = 8, 64
    r = np.linspace(0.0, 1.0, size, dtype=np.float32)
    ramp = (r[None, :] + r[:, None]) / 2.0           # diagonal gradient 0..1

    fine = cv2.resize(ramp, (big, big), interpolation=cv2.INTER_LINEAR) > 0.5
    coarse = cv2.resize((ramp > 0.5).astype(np.uint8), (big, big),
                        interpolation=cv2.INTER_NEAREST) > 0

    def edge_cols(m):
        return {int(np.argmax(row)) if row.any() else big for row in m}

    assert len(edge_cols(fine)) > len(edge_cols(coarse))   # finer: more sub-grid edge positions


# --------------------------------------------------------------------------- #
# Guided-filter edge refinement (snap the coarse mask to the real image edges)
# --------------------------------------------------------------------------- #
def test_guided_filter_follows_the_guide_edge():
    guide = np.zeros((40, 100), np.uint8)
    guide[:, 50:] = 255                                  # hard vertical image edge at col 50
    soft = cv2.GaussianBlur((guide > 0).astype(np.float32), (0, 0), 6)   # soft, spread mask edge
    out = y.guided_filter(guide, soft, radius=12, eps=1e-4)
    row = 20
    assert out[row, 10] < 0.25                           # dark side stays background
    assert out[row, 90] > 0.75                           # bright side stays player
    # the refined edge is at least as sharp as the soft input (narrower uncertain band)
    soft_band = int(((soft[row] > 0.25) & (soft[row] < 0.75)).sum())
    out_band = int(((out[row] > 0.25) & (out[row] < 0.75)).sum())
    assert out_band <= soft_band


def test_refine_edges_preserves_shape_with_downscale():
    guide = (np.random.default_rng(0).random((48, 64)) * 255).astype(np.uint8)
    prob = np.random.default_rng(1).random((48, 64)).astype(np.float32)
    out = y.refine_edges(guide, prob, radius=8, eps=1e-3, scale=2)
    assert out.shape == (48, 64)


# --------------------------------------------------------------------------- #
# letterbox geometry
# --------------------------------------------------------------------------- #
def test_letterbox_is_square_and_centred():
    frame = np.full((100, 200, 3), 7, np.uint8)      # h=100, w=200 (landscape)
    canvas, px, py, nw, nh = y._letterbox(frame, 320)
    assert canvas.shape == (320, 320, 3)
    assert (nw, nh) == (320, 160)                    # scaled by 320/200 = 1.6
    assert (px, py) == (0, 80)                        # padded top/bottom


# --------------------------------------------------------------------------- #
# BodySegmenter cadence (FakeNet, no real model)
# --------------------------------------------------------------------------- #
class FakeNet:
    def __init__(self, out0, out1):
        self.calls = 0
        self._outs = (out0, out1)

    def setInput(self, blob):
        pass

    def getUnconnectedOutLayersNames(self):
        return ("output0", "output1")

    def forward(self, names):
        self.calls += 1
        return self._outs


def _blank_outputs():
    out0 = _output0([(50, 60, 20, 40, y.PERSON_CLASS, 0.0, np.zeros(32))])  # nothing passes
    out1 = np.zeros((1, 32, 16, 16), np.float32)
    return out0, out1


def test_segment_throttles_inference_to_every_n():
    net = FakeNet(*_blank_outputs())
    seg = y.BodySegmenter(net, input_size=64, every_n=3)
    frame = np.zeros((48, 64, 3), np.uint8)
    for _ in range(6):
        out = seg.segment(frame)
    assert out.shape == (48, 64)         # mask at frame resolution
    assert net.calls == 2                # inferred on frame 0 and 3 only


def test_segment_returns_same_cached_mask_between_inferences():
    net = FakeNet(*_blank_outputs())
    seg = y.BodySegmenter(net, input_size=64, every_n=5)
    frame = np.zeros((48, 64, 3), np.uint8)
    first = seg.segment(frame)
    second = seg.segment(frame)          # skipped frame
    assert net.calls == 1
    assert second is first               # reused object, no recompute


# --------------------------------------------------------------------------- #
# Graceful fallback
# --------------------------------------------------------------------------- #
def test_create_segmenter_returns_none_without_model(monkeypatch):
    monkeypatch.setattr(y, "ensure_model",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("missing")))
    assert y.create_segmenter() is None


def test_ensure_model_raises_when_missing_and_no_download(tmp_path):
    with pytest.raises(RuntimeError):
        y.ensure_model(model_dir=str(tmp_path), download=False)
