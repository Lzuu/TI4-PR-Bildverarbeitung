"""Local 33-joint BlazePose ONNX inference and bounded temporal tracking.

Uses the existing person mask to select the player and a cached BlazePose
initializer for hip/rotation/scale. No torch, MediaPipe Python package
or network request in the frame loop.
Model layout: OpenCV Zoo pose_estimation_mediapipe (Apache-2.0).
"""
from dataclasses import dataclass
import hashlib
import logging
from pathlib import Path
import time
import urllib.request

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)
MODEL_NAME = 'pose_estimation_mediapipe_2023mar.onnx'
MODEL_SHA256 = '9d89c599319a18fb7d2e28451a883476164543182bafca5f09eb2cf767ed2f3f'
MODEL_URL = ('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
             'models/pose_estimation_mediapipe/' + MODEL_NAME)
MODEL_DIR = Path(__file__).resolve().parent / 'models'
DETECTOR_NAME = 'person_detection_mediapipe_2023mar.onnx'
DETECTOR_SHA256 = '47fd5599d6fa17608f03e0eb0ae230baa6e597d7e8a2c8199fe00abea55a701f'
DETECTOR_URL = ('https://media.githubusercontent.com/media/opencv/opencv_zoo/main/'
                'models/person_detection_mediapipe/' + DETECTOR_NAME)


def ensure_model(model_dir=MODEL_DIR, download=True, kind='landmarks'):
    """Bounded, verified, atomic first-use installation, never from update()."""
    folder = Path(model_dir)
    folder.mkdir(parents=True, exist_ok=True)
    name,expected,url = ((DETECTOR_NAME,DETECTOR_SHA256,DETECTOR_URL)
                         if kind=='detector' else (MODEL_NAME,MODEL_SHA256,MODEL_URL))
    path = folder / name
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Pose model hash mismatch: ' + str(path))
        return str(path)
    if not download:
        raise RuntimeError('Pose model missing; run python tools/prepare_suit.py')
    temp = path.with_suffix('.onnx.part')
    try:
        logger.info('Downloading pose model (%s)', name)
        with urllib.request.urlopen(url, timeout=20) as response:
            content = response.read(16 * 1024 * 1024 + 1)
        if hashlib.sha256(content).hexdigest() != expected:
            raise RuntimeError('Downloaded pose model hash mismatch')
        temp.write_bytes(content)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return str(path)


@dataclass
class Pose:
    """33 image-space xyz points; confidence combines visibility and presence."""
    points: np.ndarray
    confidence: np.ndarray

    def valid(self, index, threshold=.5):
        return (self.confidence[index] >= threshold
                and bool(np.isfinite(self.points[index]).all()))

    def segment(self, start, end):
        if not self.valid(start) or not self.valid(end):
            return None
        return self.points[[start, end], :2].copy()

    def copy(self):
        return Pose(self.points.copy(), self.confidence.copy())


def crop_transform(bbox, size=256, margin=1.15):
    """Image -> padded square ROI, preserving aspect even beyond frame edges."""
    x, y, w, h = bbox
    side = max(4., max(w, h) * margin)
    scale = size / side
    cx, cy = x + w / 2, y + h / 2
    return np.array([[scale, 0, size / 2 - scale * cx],
                     [0, scale, size / 2 - scale * cy]], np.float32)


def detector_anchors():
    # SSD feature grids of the OpenCV-Zoo BlazePose person detector, no large table.
    return np.asarray([((x+.5)/size,(y+.5)/size)
                       for size,repeats in [(28,2),(14,2),(7,6)]
                       for y in range(size) for x in range(size)
                       for _ in range(repeats)],np.float32)


def oriented_crop(hips, full_body_point):
    center=np.asarray(hips,np.float32)
    delta=np.asarray(full_body_point,np.float32)-center
    radius=max(4.,float(np.linalg.norm(delta)))
    angle=np.degrees(np.pi/2-np.arctan2(-delta[1],delta[0]))
    matrix=cv2.getRotationMatrix2D(tuple(map(float,center)),float(angle),256/(radius*2))
    matrix[:,2]+=(128-center[0],128-center[1])
    return matrix.astype(np.float32)


class PersonDetector:
    """224px BlazePose initializer, including hip/scale/rotation keypoints."""
    def __init__(self,net):
        self.net=net
        self.anchors=detector_anchors()

    def __call__(self,frame,mask):
        h,w=frame.shape[:2]
        scale=224/max(h,w)
        nw,nh=int(w*scale),int(h*scale)
        px,py=(224-nw)//2,(224-nh)//2
        canvas=np.zeros((224,224,3),np.float32)
        resized=cv2.cvtColor(cv2.resize(frame,(nw,nh)),cv2.COLOR_BGR2RGB)
        canvas[py:py+nh,px:px+nw]=resized.astype(np.float32)/127.5-1
        self.net.setInput(canvas.transpose(2,0,1)[None])
        outputs=self.net.forward(self.net.getUnconnectedOutLayersNames())
        boxes=next(x for x in outputs if x.shape[-1]==12).reshape(-1,12)
        logits=next(x for x in outputs if x.shape[-1]==1).ravel()
        scores=1/(1+np.exp(-np.clip(logits,-60,60)))
        candidates=np.flatnonzero(scores>=.5)
        if not len(candidates):return None
        # Select a detection whose estimated body circle overlaps our active mask.
        bbox=player_box(mask)
        if bbox is None:return None
        center=np.asarray(bbox[:2])+np.asarray(bbox[2:])/2
        best=None;quality=-np.inf
        for index in candidates:
            keypoints=((boxes[index,4:].reshape(4,2)/224+self.anchors[index])*max(h,w)
                       -np.array((px/scale,py/scale)))
            hip,up=keypoints[:2]
            radius=np.linalg.norm(up-hip)
            if radius<10:continue
            value=float(scores[index])-float(np.linalg.norm(hip-center)/radius)*.15
            if value>quality:best,quality=(hip,up),value
        return best


def decode_landmarks(raw, matrix):
    landmarks = np.asarray(raw, np.float32).reshape(-1, 5)[:33]
    points = landmarks[:, :3].copy()
    points[:, :2] = cv2.transform(points[None, :, :2],
                                  cv2.invertAffineTransform(matrix))[0]
    points[:, 2] /= max(float(np.linalg.norm(matrix[0, :2])), 1e-6)
    scores = 1 / (1 + np.exp(-np.clip(landmarks[:, 3:], -60, 60)))
    return Pose(points, np.minimum(scores[:, 0], scores[:, 1]))


def player_box(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < .01 * mask.size:
        return None
    return cv2.boundingRect(contour)


class PoseEstimator:
    def __init__(self, net, detector=None):
        self.net = net
        self.detector=detector
        self.failed = False
        self.last_ms = 0.
        self.reset()

    def reset(self):
        self._roi=None
        self._calls=0
        self._last_box=None

    def __call__(self, frame, mask):
        bbox = player_box(mask)
        if bbox is None or self.failed:
            return None
        started = time.perf_counter()
        try:
            if self.detector is not None:
                if self._roi is None or self._calls%5==0:
                    detected=self.detector(frame,mask)
                    if detected is not None:self._roi=detected
                if self._roi is None:return None
                if self._last_box is not None and self._calls%5!=0:
                    delta=(np.asarray(bbox[:2])+np.asarray(bbox[2:])/2
                           -np.asarray(self._last_box[:2])-np.asarray(self._last_box[2:])/2)
                    self._roi=tuple(point+delta for point in self._roi)
                matrix=oriented_crop(*self._roi)
            else:
                matrix=crop_transform(bbox)
            self._last_box=bbox
            self._calls+=1
            image=cv2.warpAffine(frame,matrix,(256,256),flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT)
            blob=cv2.cvtColor(image,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
            self.net.setInput(blob[None])  # this model uses NHWC, not blobFromImage/NCHW
            outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
            landmarks = next(x for x in outputs if x.size == 195)
            confidence = float(next(x for x in outputs if x.size == 1).ravel()[0])
            if confidence < config.POSE_CONFIDENCE:
                self._roi=None
                return None
            pose=decode_landmarks(landmarks,matrix)
            h,w=frame.shape[:2]
            outside=(pose.points[:,0]<0)|(pose.points[:,0]>=w)|(pose.points[:,1]<0)|(pose.points[:,1]>=h)
            pose.confidence[outside]=0
            return pose
        except (cv2.error, ValueError, StopIteration):
            self.failed = True
            logger.warning('Pose inference disabled; using limited suit fallback', exc_info=True)
            return None
        finally:
            self.last_ms = (time.perf_counter() - started) * 1000


class PoseTracker:
    """Adaptive smoothing + optical flow between inferences, bounded by real time.

    Only a verified network result refreshes age. Failed LK tracks lose their
    confidence, and source/body loss resets immediately. No bbox-made limbs.
    """
    def __init__(self, infer, interval=None, max_age=None):
        self.infer = infer
        self.interval = config.POSE_INTERVAL if interval is None else interval
        self.max_age = config.POSE_MAX_AGE if max_age is None else max_age
        self.reset()

    def reset(self):
        self.current = None
        self._last_verified = None
        self._last_inference = None
        self._gray = None
        self._source = None
        self._box = None
        if hasattr(self.infer,'reset'):
            self.infer.reset()

    def stabilize(self, pose, now):
        if self.current is None:
            self.current = pose.copy()
            return self.current.copy()
        fresh = pose.copy()
        valid = (fresh.confidence >= .5) & (self.current.confidence >= .5)
        movement = np.linalg.norm(fresh.points[:, :2] - self.current.points[:, :2], axis=1)
        # Tiny jitter is damped; fast movement largely passes in the same frame.
        scale = max(30., np.linalg.norm(fresh.points[11, :2] - fresh.points[12, :2]))
        alpha = np.clip(.35 + movement / scale * 1.7, .35, 1.)
        fresh.points[valid] = (self.current.points[valid] * (1 - alpha[valid, None])
                               + fresh.points[valid] * alpha[valid, None])
        self.current = fresh
        return fresh.copy()

    def _flow(self, gray):
        if self.current is None or self._gray is None:
            return
        ids = np.flatnonzero(self.current.confidence >= .5)
        if not len(ids):
            return
        points = self.current.points[ids, :2].astype(np.float32).reshape(-1, 1, 2)
        moved, ok, error = cv2.calcOpticalFlowPyrLK(
            self._gray, gray, points, None, winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, .03))
        if moved is None:
            self.current.confidence[ids] = 0
            return
        good = ok.ravel().astype(bool) & (error.ravel() < 35)
        self.current.points[ids[good], :2] = moved.reshape(-1, 2)[good]
        self.current.confidence[ids[~good]] = 0

    def update(self, frame, mask, now=None, source=None):
        now = time.monotonic() if now is None else now
        box = player_box(mask)
        if box is None:
            self.reset()
            return None
        changed = source != self._source or (self._gray is not None and self._gray.shape != mask.shape)
        if self._box is not None:
            old_center = np.array(self._box[:2]) + np.array(self._box[2:]) / 2
            center = np.array(box[:2]) + np.array(box[2:]) / 2
            changed |= np.linalg.norm(center - old_center) > max(box[2:]) * .65
        if changed:
            self.reset()
        self._source, self._box = source, box
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._last_inference is not None and now - self._last_inference < self.interval:
            self._flow(gray)
        else:
            self._last_inference = now
            measured = self.infer(frame, mask) if self.infer is not None else None
            if measured is not None:
                self.stabilize(measured, now)
                self._last_verified = now
        self._gray = gray
        if self._last_verified is None or now - self._last_verified > self.max_age:
            self.current = None
        return None if self.current is None else self.current.copy()


def create_tracker(download=True):
    if not config.POSE_ENABLED:
        return PoseTracker(None)
    try:
        detector=PersonDetector(cv2.dnn.readNetFromONNX(ensure_model(download=download,kind='detector')))
        estimator = PoseEstimator(cv2.dnn.readNetFromONNX(ensure_model(download=download)),detector)
        logger.info('Pose model ready: BlazePose 33 joints, 256px, interval %.3fs', config.POSE_INTERVAL)
        return PoseTracker(estimator)
    except (OSError, RuntimeError, cv2.error):
        logger.warning('Pose unavailable; segmented cloth + face helmet only', exc_info=True)
        return PoseTracker(None)
