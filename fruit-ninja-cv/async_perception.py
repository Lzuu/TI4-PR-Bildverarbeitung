"""Non-blocking adapters; model inference belongs exclusively to the worker."""
import time
import config
import pose_tracking
import yolo_seg


class AsyncSegmenter:
    IDLE, LOADING, READY, FAILED = 'idle', 'loading', 'ready', 'failed'

    def __init__(self, worker):
        self.worker = worker
        self.model = None
        self.state = self.IDLE
        self._count = self._generation = 0
        self._seen = self._loaded_generation = None
        self.mask_age = 0.
        self.last_inference_seconds = None

    def _infer(self, frame, generation):
        if self._loaded_generation != generation:
            self.model = yolo_seg.create_segmenter()
            self._loaded_generation = generation
        if self.model is None:
            raise RuntimeError('YOLO model unavailable')
        return self.model._infer(frame)

    def request(self):
        if self.state == self.IDLE:
            self.state = self.LOADING

    def get(self):
        return self if self.state == self.READY else None

    @property
    def failed(self):
        return self.worker.failed('yolo')

    def segment(self, frame, source=None, captured_at=None):
        now = time.monotonic()
        captured_at = now if captured_at is None else captured_at
        generation = self._generation
        key = (source, frame.shape, generation)
        self.worker.activate('yolo', key)
        result = self.worker.poll('yolo', key)
        self.last_inference_seconds = None
        if result is not None:
            self.state = self.READY
            self.mask_age = max(0., now-result.captured_at)
            if result.sequence != self._seen:
                self.last_inference_seconds = result.seconds
                self._seen = result.sequence
        due = self._count % max(1, config.YOLO_EVERY_N) == 0 or result is None
        self._count += 1
        self.worker.submit('yolo', key, frame, lambda image: self._infer(image, generation), captured_at, due=due)
        if self.failed:
            self.state = self.FAILED
        if result is None or self.mask_age > config.ASYNC_MAX_MASK_AGE:
            return None
        return result.value

    def mark_failed(self):
        self.state = self.FAILED

    def retry(self):
        if self.failed:
            self._generation += 1
            self.worker.retry('yolo')
            self.state = self.IDLE


class AsyncPoseInfer:
    def __init__(self, worker):
        self.worker = worker
        self.model = None
        self._loaded = False
        self._model_key = None
        self.generation = 0
        self._last_request = self._seen = None
        self.last_inference_seconds = None

    def reset(self):
        self.generation += 1
        self._last_request = self._seen = None
        self.worker.activate('pose', ('reset', self.generation))

    def poll(self, frame, mask, now, source, captured_at=None):
        key = (source, frame.shape, self.generation)
        self.worker.activate('pose', key)
        result = self.worker.poll('pose', key)
        due = self._last_request is None or now-self._last_request >= config.POSE_INTERVAL
        mask_copy = mask.copy()
        def infer(image):
            if not self._loaded:
                self.model = pose_tracking.create_tracker().infer
                self._loaded = True
            if self.model is None:
                raise RuntimeError('Pose model unavailable')
            if key != self._model_key:
                self.model.reset()
                self._model_key = key
            value = self.model(image, mask_copy)
            if self.model.failed:
                raise RuntimeError('Pose model failed')
            return value
        if config.POSE_ENABLED:
            self.worker.submit('pose', key, frame, infer, now if captured_at is None else captured_at, due=due)
        if due:
            self._last_request = now
        self.last_inference_seconds = None
        if result is None or result.sequence == self._seen:
            return None
        self._seen = result.sequence
        self.last_inference_seconds = result.seconds
        return result
