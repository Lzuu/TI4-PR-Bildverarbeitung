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


class Camera:
    """The camera in use; can be switched to another index at runtime.

    A camera only counts as open once it delivered a picture. If a switch fails
    and the previous camera cannot be reopened either, there is no camera:
    ``read`` then reports failure instead of crashing (#41, CR-02).
    """

    def __init__(self, opener=default_opener):
        self._opener = opener
        self.cap = None
        self.index = None

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
                self.cap, self.index = cap, index
                return True
            logger.info("Camera %d not available", index)
        return False

    def switch(self, index):
        """Change to camera ``index``; keeps the current one if that fails."""
        if index == self.index:
            return True
        old_cap, old_index = self.cap, self.index
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
            return False
        self.cap, self.index = cap, index
        logger.info("Switched camera %d -> %d", old_index, index)
        return True

    @property
    def is_open(self):
        return self.cap is not None

    def read(self):
        if self.cap is None:
            return False, None       # no camera: the game loop counts failed reads
        return self.cap.read()

    def release(self):
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
