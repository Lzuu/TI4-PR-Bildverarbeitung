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


# -- CR-09 (#41): YOLO failures switch the optional feature off, no crash ----
class FailingNet(FakeNet):
    def forward(self, names):
        self.calls += 1
        raise cv2.error("inference failed")


def test_create_segmenter_returns_none_on_file_errors(monkeypatch):
    def denied(*a, **k):
        raise PermissionError("model folder not readable")
    monkeypatch.setattr(y, "ensure_model", denied)
    assert y.create_segmenter() is None


def test_inference_error_disables_the_segmenter_once(caplog):
    net = FailingNet(*_blank_outputs())
    seg = y.BodySegmenter(net, input_size=64, every_n=1)
    frame = np.zeros((48, 64, 3), np.uint8)
    assert seg.segment(frame) is None                      # no exception
    assert seg.failed
    assert seg.segment(frame) is None                      # no retry every frame
    assert net.calls == 1
    errors = [r for r in caplog.records if r.levelname in ("ERROR", "WARNING")]
    assert len(errors) == 1                                # one summarised log entry


# -- CR-10 (#41): mask cache is bound to the camera / mode session -----------
def test_new_source_gets_a_fresh_inference_despite_the_same_size():
    net = FakeNet(*_blank_outputs())
    seg = y.BodySegmenter(net, input_size=64, every_n=10)
    frame = np.zeros((48, 64, 3), np.uint8)
    seg.segment(frame, source=("cam", 0))
    seg.segment(frame, source=("cam", 0))                  # same source: cached
    assert net.calls == 1
    seg.segment(frame, source=("cam", 1))                  # camera switched
    assert net.calls == 2


def test_cadence_for_an_unchanged_source_stays_the_same():
    net = FakeNet(*_blank_outputs())
    seg = y.BodySegmenter(net, input_size=64, every_n=3)
    frame = np.zeros((48, 64, 3), np.uint8)
    for _ in range(6):
        seg.segment(frame, source=("cam", 0))
    assert net.calls == 2


# -- PERF-07 (#42): load YOLO only when it is used, in the background --------
import io
import os

import pytest

import config


class ManualStart:
    """Collects the loader jobs instead of starting threads; run them by hand."""

    def __init__(self):
        self.jobs = []

    def __call__(self, job):
        self.jobs.append(job)

    def run_all(self):
        jobs, self.jobs = self.jobs, []
        for job in jobs:
            job()


def _counting_factory(result):
    calls = []

    def factory():
        calls.append(1)
        return result
    return factory, calls


def test_lazy_segmenter_loads_nothing_until_requested():
    factory, calls = _counting_factory(object())
    lazy = y.LazySegmenter(factory, start=ManualStart())
    assert lazy.state == y.LazySegmenter.IDLE
    assert lazy.get() is None
    assert calls == []


def test_lazy_segmenter_loads_in_the_background_once():
    seg = object()
    factory, calls = _counting_factory(seg)
    start = ManualStart()
    lazy = y.LazySegmenter(factory, start=start)
    lazy.request()
    lazy.request()                                   # still loading: no second job
    assert lazy.state == y.LazySegmenter.LOADING and lazy.get() is None
    assert len(start.jobs) == 1
    start.run_all()
    assert lazy.state == y.LazySegmenter.READY and lazy.get() is seg
    lazy.request()                                   # ready: reused, not reloaded
    assert calls == [1] and start.jobs == []


def test_lazy_segmenter_failure_is_kept_until_a_deliberate_retry():
    factory, calls = _counting_factory(None)         # create_segmenter: None = unavailable
    start = ManualStart()
    lazy = y.LazySegmenter(factory, start=start)
    lazy.request()
    start.run_all()
    assert lazy.state == y.LazySegmenter.FAILED
    for _ in range(5):
        lazy.request()                               # no retry per frame
    assert start.jobs == [] and calls == [1]
    lazy.retry()
    lazy.request()
    start.run_all()
    assert calls == [1, 1]


def test_lazy_segmenter_unexpected_error_counts_as_failed():
    def broken():
        raise ValueError("bad model")
    start = ManualStart()
    lazy = y.LazySegmenter(broken, start=start)
    lazy.request()
    start.run_all()
    assert lazy.state == y.LazySegmenter.FAILED


def test_inference_failure_marks_the_loader_failed():
    seg = y.BodySegmenter(FailingNet(*_blank_outputs()), input_size=64)
    start = ManualStart()
    lazy = y.LazySegmenter(lambda: seg, start=start)
    lazy.request()
    start.run_all()
    lazy.get().segment(np.zeros((48, 64, 3), np.uint8))
    assert lazy.get() is None and lazy.state == y.LazySegmenter.FAILED


def test_download_uses_a_timeout(monkeypatch, tmp_path):
    seen = {}
    payload = b"model-bytes"

    def fake_urlopen(url, timeout=None):
        seen["timeout"] = timeout
        return io.BytesIO(payload)
    monkeypatch.setattr(y.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(y, "MODEL", y.ModelFile("m.onnx", "http://example.invalid/m",
                                                y.hashlib.sha256(payload).hexdigest()))
    path = y.ensure_model(str(tmp_path))
    assert seen["timeout"] == config.YOLO_DOWNLOAD_TIMEOUT
    assert open(path, "rb").read() == payload


def test_download_timeout_is_a_clean_failure(monkeypatch, tmp_path):
    def slow(url, timeout=None):
        raise TimeoutError("timed out")
    monkeypatch.setattr(y.urllib.request, "urlopen", slow)
    with pytest.raises(RuntimeError):
        y.ensure_model(str(tmp_path))
    assert os.listdir(tmp_path) == []                # no half-written .part file left


# -- PERF-08 (#42): mask age and inference time are visible and budgeted -----
class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def _seg(clock, every_n=3, max_age=None):
    return y.BodySegmenter(FakeNet(*_blank_outputs()), input_size=64, every_n=every_n,
                           max_mask_age=max_age, clock=clock)


def test_age_is_zero_on_an_inference_frame_and_grows_on_cache_frames():
    clock = Clock()
    seg = _seg(clock, every_n=3, max_age=10.0)
    frame = np.zeros((48, 64, 3), np.uint8)
    seg.segment(frame)
    assert seg.mask_age == 0.0 and seg.inferred
    clock.t += 0.033
    seg.segment(frame)
    assert not seg.inferred
    assert seg.mask_age == pytest.approx(0.033)
    clock.t += 0.033
    seg.segment(frame)
    assert seg.mask_age == pytest.approx(0.066)
    seg.segment(frame)                                    # 4th frame: inference again
    assert seg.inferred and seg.mask_age == 0.0


def test_inference_time_is_reported_only_for_inference_frames():
    clock = Clock()
    seg = _seg(clock, every_n=2, max_age=10.0)
    frame = np.zeros((48, 64, 3), np.uint8)
    seg.segment(frame)
    assert seg.last_inference_seconds is not None
    seg.segment(frame)
    assert seg.last_inference_seconds is None             # cache hit: nothing to time


def test_old_mask_is_refreshed_early_at_a_low_frame_rate():
    clock = Clock()
    seg = _seg(clock, every_n=3, max_age=0.1)
    frame = np.zeros((48, 64, 3), np.uint8)
    for _ in range(6):                                    # 10 fps: 0.1 s per frame
        seg.segment(frame)
        clock.t += 0.1
    assert seg.net.calls > 2                              # more often than every 3rd frame


def test_cadence_is_unchanged_at_a_fast_frame_rate():
    clock = Clock()
    seg = _seg(clock, every_n=3, max_age=0.1)
    frame = np.zeros((48, 64, 3), np.uint8)
    for _ in range(6):                                    # 60 fps: mask stays young
        seg.segment(frame)
        clock.t += 1 / 60
    assert seg.net.calls == 2


def test_mask_age_budget_defaults_to_config():
    seg = y.BodySegmenter(FakeNet(*_blank_outputs()), input_size=64)
    assert seg.max_mask_age == config.YOLO_MAX_MASK_AGE


def test_no_early_refresh_when_inference_itself_is_expensive():
    # on a slow machine an extra inference per frame would only make every frame
    # slower (feedback loop) -- the plain cadence stays
    clock = Clock()

    class SlowNet(FakeNet):
        def forward(self, names):
            clock.t += 0.07                               # 70 ms per inference
            return super().forward(names)
    seg = y.BodySegmenter(SlowNet(*_blank_outputs()), input_size=64, every_n=3,
                          max_mask_age=0.1, clock=clock)
    frame = np.zeros((48, 64, 3), np.uint8)
    for _ in range(6):
        seg.segment(frame)
        clock.t += 0.1                                    # slow frames: mask always > budget
    assert seg.net.calls == 2                             # frames 0 and 3 only
    assert seg.last_inference_seconds is None or seg.last_inference_seconds == pytest.approx(0.07)
