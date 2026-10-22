"""Body segmentation with YOLOv8-seg, run through ``cv2.dnn`` (no torch).

Used by the ``"yolo"`` background mode: a person-segmentation mask cuts the
player out of *any* background (no green screen) and feeds the suit. It mirrors
``face_id.py`` -- the ONNX model is downloaded into ``models/`` on first use and
checked against a SHA256 hash; if it cannot be loaded the feature degrades
gracefully (``create_segmenter`` returns ``None`` and the mode falls back).

The raw YOLOv8-seg outputs are decoded in pure functions so the maths is
unit-tested headlessly (no model, no GPU):
  * ``output0`` ``(1, 116, A)`` -- per anchor: 4 box (cx,cy,w,h in input px),
    80 class scores (person = class 0), 32 mask coefficients.
  * ``output1`` ``(1, 32, gh, gw)`` -- 32 mask prototypes. A detection's mask is
    ``sigmoid(coeffs @ prototypes)``, cropped to its box.

The model (``yolov8n-seg.onnx``) is exported from Ultralytics YOLOv8n-seg
(AGPL-3.0) and hosted as a release asset for runtime download.
"""

import hashlib
import logging
import os
import urllib.request
from collections import namedtuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
PERSON_CLASS = 0                 # COCO class index for "person"
NUM_CLASSES = 80
NMS_IOU = 0.5

ModelFile = namedtuple("ModelFile", "filename url sha256")
MODEL = ModelFile(
    "yolov8n-seg.onnx",
    "https://github.com/Lzuu/TI4-PR-Bildverarbeitung/releases/download/models-v1/yolov8n-seg.onnx",
    "42380ea8b488c350cc76b6f43690c804089f4876fde393a5b3a86cf4736d47b3")


# --------------------------------------------------------------------------- #
# Model download (same pattern as face_id.ensure_models)
# --------------------------------------------------------------------------- #
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_model(model_dir=MODEL_DIR, download=True):
    """Return the local ONNX path, downloading it on first use.

    Raises ``RuntimeError`` if the model is missing (and may not be downloaded),
    the download fails or the file does not match its hash.
    """
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, MODEL.filename)
    if not os.path.exists(path):
        if not download:
            raise RuntimeError(f"{MODEL.filename} fehlt in {model_dir}")
        logger.info("Downloading %s ...", MODEL.filename)
        tmp = path + ".part"
        try:
            urllib.request.urlretrieve(MODEL.url, tmp)
        except OSError as exc:
            raise RuntimeError(f"Download von {MODEL.filename} fehlgeschlagen: {exc}")
        os.replace(tmp, path)
    if _sha256(path) != MODEL.sha256:
        raise RuntimeError(f"{MODEL.filename} ist beschaedigt (Hash stimmt nicht)")
    return path


# --------------------------------------------------------------------------- #
# Decode (pure, headless-testable)
# --------------------------------------------------------------------------- #
def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _letterbox(frame, size):
    """Resize ``frame`` into a ``size``x``size`` square, keeping aspect (grey pad).

    Returns ``(canvas, pad_x, pad_y, new_w, new_h)`` so the mask can later be
    un-padded and scaled back to the original frame.
    """
    h, w = frame.shape[:2]
    s = size / max(h, w)
    nw, nh = int(round(w * s)), int(round(h * s))
    canvas = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(frame, (nw, nh))
    return canvas, px, py, nw, nh


def person_detections(output0, conf):
    """Person boxes above ``conf`` from raw ``output0`` ``(1, 4+80+32, A)``.

    Returns ``(xywh, confs, coeffs)``: top-left boxes (input px), confidences and
    the 32 mask coefficients per kept anchor.
    """
    p = output0[0].T                                  # (A, 116)
    scores = p[:, 4 + PERSON_CLASS]                   # person is class 0
    keep = scores >= conf
    boxes = p[keep, :4]
    coeffs = p[keep, 4 + NUM_CLASSES:]
    confs = scores[keep]
    cx, cy, bw, bh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    xywh = np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1)
    return xywh, confs, coeffs


def nms(xywh, confs, conf, iou=NMS_IOU):
    """Indices surviving non-maximum suppression (wraps ``cv2.dnn.NMSBoxes``)."""
    if len(xywh) == 0:
        return []
    idx = cv2.dnn.NMSBoxes(xywh.tolist(), confs.tolist(), conf, iou)
    return list(np.array(idx).flatten()) if len(idx) else []


def person_mask(coeffs, protos, box, size, thresh):
    """One detection's 0/255 mask at ``size``x``size``, cropped to its ``box``."""
    gh, gw = protos.shape[1:]
    m = _sigmoid(coeffs @ protos.reshape(protos.shape[0], -1)).reshape(gh, gw)
    m = cv2.resize(m, (size, size))
    box_mask = np.zeros((size, size), np.uint8)
    x, y, w, h = box
    box_mask[max(0, int(y)):int(y + h), max(0, int(x)):int(x + w)] = 1
    return (((m > thresh) & (box_mask > 0)).astype(np.uint8)) * 255


def assemble_mask(xywh, coeffs, idx, protos, size, thresh):
    """Union (0/255) of the person masks for the kept detections ``idx``."""
    full = np.zeros((size, size), np.uint8)
    for i in idx:
        full = np.maximum(full, person_mask(coeffs[i], protos, xywh[i], size, thresh))
    return full


# --------------------------------------------------------------------------- #
# Segmenter
# --------------------------------------------------------------------------- #
class BodySegmenter:
    """Produces a 0/255 player mask from a BGR frame via YOLOv8-seg.

    Inference is throttled to every ``every_n``-th frame (it costs ~15 ms); the
    last mask is reused in between, which is fine for an overlay that moves
    slowly relative to the frame rate.
    """

    def __init__(self, net, input_size=None, conf=None, mask_thresh=None, every_n=None):
        self.net = net
        self.input = input_size or config.YOLO_INPUT
        self.conf = config.YOLO_CONF if conf is None else conf
        self.mask_thresh = config.YOLO_MASK_THRESH if mask_thresh is None else mask_thresh
        self.every_n = max(1, config.YOLO_EVERY_N if every_n is None else every_n)
        self._counter = 0
        self._last = None

    def _infer(self, frame):
        canvas, px, py, nw, nh = _letterbox(frame, self.input)
        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0, (self.input, self.input),
                                     swapRB=True)
        self.net.setInput(blob)
        out0, out1 = self.net.forward(self.net.getUnconnectedOutLayersNames())
        xywh, confs, coeffs = person_detections(out0, self.conf)
        idx = nms(xywh, confs, self.conf)
        small = assemble_mask(xywh, coeffs, idx, out1[0], self.input, self.mask_thresh)
        h, w = frame.shape[:2]
        crop = small[py:py + nh, px:px + nw]           # drop the letterbox padding
        return cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)

    def segment(self, frame):
        """Return the 0/255 player mask for ``frame`` (throttled; reuses last)."""
        run = (self._counter % self.every_n == 0
               or self._last is None
               or self._last.shape != frame.shape[:2])
        self._counter += 1
        if run:
            self._last = self._infer(frame)
        return self._last


def create_segmenter():
    """A ``BodySegmenter``, or ``None`` if the model is unavailable (feature off)."""
    try:
        path = ensure_model()
        net = cv2.dnn.readNetFromONNX(path)
    except (RuntimeError, cv2.error):
        logger.warning("Body segmentation (YOLO) disabled -- model unavailable",
                       exc_info=True)
        return None
    logger.info("Body segmentation ready (YOLOv8-seg, %dpx)", config.YOLO_INPUT)
    return BodySegmenter(net)
