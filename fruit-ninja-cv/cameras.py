"""Camera discovery, selection and opening.

OpenCV cannot list cameras or tell their names, so cameras are found by trying
the indices 0..CAMERA_MAX_INDEX-1 -- once at startup, *before* the game camera
is opened (about 1 s per camera). Probing while another camera streams is not
reliable: e.g. front and rear camera of a laptop often share one controller
that serves only one stream, and the running camera stops delivering frames.

The chosen camera is stored in ``config.SETTINGS_FILE`` and used again on the
next start; if it is gone, another available one is used.
"""

import json
import logging
import sys
import threading
import time

import cv2

import config

logger = logging.getLogger(__name__)

PREVIEW_SIZE = (192, 108)        # Thumbnail size (w, h) for the selection panel


def backend():
    # On Windows, MSMF delivers 1080p MJPG at 30 fps; DirectShow ignores MJPG
    # (5 fps at 1080p). Elsewhere let OpenCV pick (AVFoundation on macOS).
    return cv2.CAP_MSMF if sys.platform == "win32" else cv2.CAP_ANY


def default_opener(index):
    return cv2.VideoCapture(index, backend())


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def probe(indices, opener=default_opener):
    """``{index: preview}`` for every index that opens and delivers a frame."""
    found = {}
    for index in indices:
        cap = opener(index)
        try:
            if not cap.isOpened():
                continue
            ok, frame = cap.read()
            if ok and frame is not None:
                found[index] = cv2.resize(frame, PREVIEW_SIZE)
        finally:
            cap.release()
    return found


# --------------------------------------------------------------------------- #
# Opening and switching
# --------------------------------------------------------------------------- #
def candidate_order(saved, default, max_index=config.CAMERA_MAX_INDEX):
    """Indices to try on startup: saved choice, then the default, then the rest."""
    order = []
    for index in [saved, default, *range(max_index)]:
        if index is not None and index not in order:
            order.append(index)
    return order


def choose_index(saved, default, available):
    """Saved camera if it was found, else the default, else any found one (or None)."""
    for index in (saved, default):
        if index is not None and index in available:
            return index
    return min(available) if available else None


FIRST_FRAME_TRIES = 5            # reads to wait for the first picture after opening


class LatestFrameReader:
    """Reads ``cap`` in its own thread and keeps only the newest picture (#60, PERF-13).

    ``cap.read()`` blocks for the next picture and decodes the 1080p MJPG; OpenCV
    releases the GIL meanwhile, so the game loop keeps running. ``read()`` returns
    the newest picture not handed out yet -- older ones are dropped, so a slow loop
    always shows the current camera image -- or waits up to ``timeout`` for one.
    """

    def __init__(self, cap, timeout=1.0):
        self._cap = cap
        self._timeout = timeout
        self._cond = threading.Condition()
        self._latest = (False, None)
        self._count = 0              # pictures read so far
        self._taken = 0              # value of _count at the last read()
        self._running = True
        self._thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self._thread.start()

    def _run(self):
        while self._running:
            ok, image = self._cap.read()
            with self._cond:
                if not self._running:
                    return
                self._latest = (ok, image if ok else None)
                self._count += 1
                self._cond.notify_all()
            if not ok:
                time.sleep(0.01)     # a failing camera must not spin the CPU

    def read(self):
        with self._cond:
            if not self._cond.wait_for(lambda: self._count != self._taken, self._timeout):
                return False, None   # no new picture in time: counts as a failed read
            self._taken = self._count
            return self._latest

    def stop(self):
        """End the thread; returns once ``cap`` is no longer read (then it may be released)."""
        with self._cond:
            self._running = False
            self._cond.notify_all()
        self._thread.join(timeout=2.0)
        if self._thread.is_alive():
            logger.warning("Camera reader did not stop within 2 s")


class Camera:
    """The camera in use; can be switched to another index at runtime.

    A camera only counts as open once it delivered a picture. If a switch fails
    and the previous camera cannot be reopened either, there is no camera:
    ``read`` then reports failure instead of crashing (#41, CR-02).
    """

    def __init__(self, opener=default_opener, threaded=False):
        self._opener = opener
        self._threaded = threaded    # read in a LatestFrameReader thread (#60)
        self._reader = None
        self.cap = None
        self.index = None
        self.last_index = None       # the camera that worked last (for reopen)

    def _start_reader(self):
        if self._threaded and self.cap is not None:
            self._reader = LatestFrameReader(self.cap)

    def _stop_reader(self):
        if self._reader is not None:
            self._reader.stop()      # nobody reads the capture once this returns
            self._reader = None

    def _open(self, index):
        cap = self._opener(index)
        if not cap.isOpened():
            cap.release()
            return None
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FPS, 30)
        for _ in range(FIRST_FRAME_TRIES):           # opened is not enough: needs a picture
            ok, img = cap.read()
            if ok and img is not None:
                break
        else:
            logger.warning("Camera %d opens but delivers no picture -- skipped", index)
            cap.release()
            return None
        logger.info("Opened camera %d: %dx%d @ %.0f fps", index,
                    cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                    cap.get(cv2.CAP_PROP_FPS))
        return cap

    def open_first(self, candidates):
        """Open the first camera that works. Returns False if none does."""
        for index in candidates:
            cap = self._open(index)
            if cap is not None:
                self.cap, self.index, self.last_index = cap, index, index
                self._start_reader()
                return True
            logger.info("Camera %d not available", index)
        return False

    def reopen(self):
        """Get a camera back after a switch and its rollback both failed (#69).

        Tries the last working camera first, then the configured one, then the rest.
        Returns True if a camera is open (also if one already was).
        """
        if self.cap is not None:
            return True
        return self.open_first(candidate_order(self.last_index, config.CAMERA_INDEX))

    def switch(self, index):
        """Change to camera ``index``; keeps the current one if that fails."""
        if index == self.index:
            return True
        old_cap, old_index = self.cap, self.index
        self._stop_reader()
        if old_cap is not None:
            old_cap.release()        # some drivers do not allow two open streams
        cap = self._open(index)
        if cap is None:
            logger.warning("Switching to camera %d failed, reopening %s", index, old_index)
            self.cap = self._open(old_index) if old_index is not None else None
            if self.cap is None:
                logger.error("Reopening camera %s failed as well -- no camera available",
                             old_index)
                self.index = None
            self._start_reader()
            return False
        self.cap, self.index, self.last_index = cap, index, index
        self._start_reader()
        logger.info("Switched camera %d -> %d", old_index, index)
        return True

    @property
    def is_open(self):
        return self.cap is not None

    def read(self):
        if self.cap is None:
            return False, None       # no camera: the game loop counts failed reads
        if self._reader is not None:
            return self._reader.read()
        return self.cap.read()

    def release(self):
        self._stop_reader()
        if self.cap is not None:
            self.cap.release()


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def _load_settings(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _update_setting(path, key, value):
    settings = _load_settings(path)
    settings[key] = value
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings, f)
    except OSError:
        logger.warning("Could not write settings file: %s", path)


def load_camera_index(path=config.SETTINGS_FILE):
    value = _load_settings(path).get("camera_index")
    return value if isinstance(value, int) else None


def save_camera_index(path, index):
    _update_setting(path, "camera_index", index)


def load_background_mode(path=config.SETTINGS_FILE):
    value = _load_settings(path).get("background_mode")
    return value if isinstance(value, str) else None


def save_background_mode(path, mode):
    _update_setting(path, "background_mode", mode)
