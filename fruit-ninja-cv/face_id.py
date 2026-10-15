"""Player identification by face.

Detection uses YuNet (``cv2.FaceDetectorYN``), recognition uses SFace
(``cv2.FaceRecognizerSF``): every face is aligned to a 112x112 crop and turned
into a feature vector. A player is recognised if the cosine similarity to one
of their reference images reaches ``config.FACE_MATCH_THRESHOLD``; everyone
else plays as "Gast".

Reference images live in ``faces/<Name>/`` (record them with ``enroll.py``).
They are biometric data and are therefore not committed to git.

Both models come from the OpenCV Model Zoo and are downloaded on first use
into ``models/`` (checked against a SHA256 hash).
"""

import hashlib
import logging
import os
import time
import urllib.request
from collections import Counter, namedtuple

import cv2
import numpy as np

import config

logger = logging.getLogger(__name__)

GUEST = "Gast"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
ALIGNED_SIZE = 112               # SFace works on aligned 112x112 face crops

ModelFile = namedtuple("ModelFile", "filename url sha256")
_ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models/"
MODELS = {
    "detector": ModelFile(
        "face_detection_yunet_2023mar.onnx",
        _ZOO + "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "recognizer": ModelFile(
        "face_recognition_sface_2021dec.onnx",
        _ZOO + "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
}


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_models(model_dir=MODEL_DIR, download=True):
    """Return ``{"detector": path, "recognizer": path}``, downloading if needed.

    Raises ``RuntimeError`` if a model is missing (and may not be downloaded),
    the download fails or a file does not match its hash.
    """
    os.makedirs(model_dir, exist_ok=True)
    paths = {}
    for key, model in MODELS.items():
        path = os.path.join(model_dir, model.filename)
        if not os.path.exists(path):
            if not download:
                raise RuntimeError(f"{model.filename} fehlt in {model_dir}")
            logger.info("Downloading %s ...", model.filename)
            tmp = path + ".part"
            try:
                urllib.request.urlretrieve(model.url, tmp)
            except OSError as exc:
                raise RuntimeError(f"Download von {model.filename} fehlgeschlagen: {exc}")
            os.replace(tmp, path)
        if _sha256(path) != model.sha256:
            raise RuntimeError(f"{model.filename} ist beschaedigt (Hash stimmt nicht)")
        paths[key] = path
    return paths


# --------------------------------------------------------------------------- #
# Pure helpers (no models needed)
# --------------------------------------------------------------------------- #
def best_match(feature, gallery, threshold):
    """Best matching person for a normalised feature vector.

    ``gallery`` maps a name to a list of normalised reference features.
    Returns ``(name, similarity)``; below ``threshold`` the name is ``GUEST``.
    """
    best_name, best_score = GUEST, 0.0
    for name, refs in gallery.items():
        for ref in refs:
            score = float(np.dot(feature.ravel(), ref.ravel()))
            if score > best_score:
                best_name, best_score = name, score
    if best_score < threshold:
        return GUEST, best_score
    return best_name, best_score


def display_name(name):
    """Name for the OpenCV HUD, which cannot render umlauts."""
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("Ä", "Ae"), ("Ö", "Oe"),
                 ("Ü", "Ue"), ("ß", "ss")):
        name = name.replace(a, b)
    return name


def person_dir(faces_dir, name):
    """Folder for a person's reference images; rejects names that leave ``faces_dir``."""
    name = name.strip()
    if not name or name in (".", "..") or any(c in name for c in "/\\:"):
        raise ValueError(f"Ungueltiger Name: {name!r}")
    return os.path.join(faces_dir, name)


class PlayerVote:
    """Decide on a player over several frames instead of trusting a single one.

    ``update(name)`` gets the per-frame result (``None`` = no face) and returns
    the decided name once a name was seen ``needed`` times, or after
    ``timeout`` seconds the most frequent name (``GUEST`` if there was none).
    """

    def __init__(self, needed=config.FACE_VOTES_NEEDED,
                 timeout=config.FACE_TIMEOUT_SECONDS, time_fn=time.time):
        self.needed = needed
        self.timeout = timeout
        self._time = time_fn
        self.reset()

    def reset(self):
        self.votes = Counter()
        self._start = self._time()

    def progress(self):
        """Elapsed share of the timeout (0..1), for a progress bar."""
        return min((self._time() - self._start) / self.timeout, 1.0)

    def update(self, name):
        if name is not None:
            self.votes[name] += 1
            if self.votes[name] >= self.needed:
                return name
        if self._time() - self._start >= self.timeout:
            return self.votes.most_common(1)[0][0] if self.votes else GUEST
        return None


# --------------------------------------------------------------------------- #
# Detection + recognition
# --------------------------------------------------------------------------- #
class FaceIdentifier:
    def __init__(self, detector_path, recognizer_path,
                 threshold=config.FACE_MATCH_THRESHOLD,
                 score_threshold=config.FACE_DETECT_SCORE):
        self._detector = cv2.FaceDetectorYN.create(detector_path, "", (320, 320),
                                                   score_threshold)
        self._recognizer = cv2.FaceRecognizerSF.create(recognizer_path, "")
        self.threshold = threshold
        self.gallery = {}        # name -> list of normalised features

    @property
    def persons(self):
        return sorted(self.gallery)

    def detect(self, img):
        """All faces in ``img`` (YuNet rows: x, y, w, h, 5 landmarks, score), largest first."""
        self._detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self._detector.detect(img)
        if faces is None:
            return []
        return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)

    def aligned_crop(self, img, face):
        return self._recognizer.alignCrop(img, face)

    def _feature_of_crop(self, crop):
        f = self._recognizer.feature(crop).ravel()
        return f / np.linalg.norm(f)

    def feature(self, img, face):
        return self._feature_of_crop(self.aligned_crop(img, face))

    def match(self, img, face):
        """``(name, similarity)`` for one detected face."""
        return best_match(self.feature(img, face), self.gallery, self.threshold)

    def add_person(self, name, images):
        """Add reference images (photos or aligned 112x112 crops). Returns features added."""
        added = 0
        for img in images:
            if img is None:
                continue
            if img.shape[:2] == (ALIGNED_SIZE, ALIGNED_SIZE):
                feature = self._feature_of_crop(img)
            else:
                faces = self.detect(img)
                if not faces:
                    continue
                feature = self.feature(img, faces[0])
            self.gallery.setdefault(name, []).append(feature)
            added += 1
        return added

    def load_gallery(self, faces_dir):
        """Load all ``faces_dir/<Name>/`` folders. Returns the number of persons."""
        if not os.path.isdir(faces_dir):
            return 0
        for name in sorted(os.listdir(faces_dir)):
            folder = os.path.join(faces_dir, name)
            if not os.path.isdir(folder):
                continue
            files = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                     if f.lower().endswith(IMAGE_EXTENSIONS)]
            added = self.add_person(name, [cv2.imread(f) for f in files])
            if added:
                logger.info("Player %s: %d reference image(s)", name, added)
            else:
                logger.warning("Player %s: no usable face in %s", name, folder)
        return len(self.gallery)

    def identify(self, frame):
        """``(name, similarity, box)`` for the largest face, or ``(None, 0.0, None)``.

        The frame is downscaled for speed; ``box`` is (x, y, w, h) in frame coordinates.
        """
        scale = min(1.0, config.FACE_INPUT_WIDTH / frame.shape[1])
        small = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale,
                                                      interpolation=cv2.INTER_AREA)
        faces = self.detect(small)
        if not faces:
            return None, 0.0, None
        name, score = self.match(small, faces[0])
        box = tuple(int(round(v / scale)) for v in faces[0][:4])
        return name, score, box


def create_identifier(faces_dir=None):
    """Identifier with the enrolled players, or ``None`` if the models are unavailable."""
    faces_dir = faces_dir or os.path.join(BASE_DIR, config.FACES_DIR)
    try:
        paths = ensure_models()
        identifier = FaceIdentifier(paths["detector"], paths["recognizer"])
    except (RuntimeError, cv2.error):
        logger.warning("Face identification disabled -- everyone plays as %s", GUEST,
                       exc_info=True)
        return None
    count = identifier.load_gallery(faces_dir)
    logger.info("Face identification ready: %d enrolled player(s) in %s", count, faces_dir)
    return identifier
