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


def face_in_circle(box, circle):
    """Is the face (x, y, w, h) centred in the scan circle and close enough to the camera?"""
    if box is None:
        return False
    x, y, w, h = box
    cx, cy, r = circle
    fx, fy = x + w / 2, y + h / 2
    centred = (fx - cx) ** 2 + (fy - cy) ** 2 <= (0.5 * r) ** 2
    return centred and 0.4 * r <= w <= 2.0 * r


def default_player_name(existing):
    """'Spieler N' with the lowest number not taken yet."""
    n = 1
    while f"Spieler {n}" in existing:
        n += 1
    return f"Spieler {n}"


class FaceScanner:
    """Decides what a face in the scan circle is: a known player or a new one.

    ``update(in_circle, name, crop)`` gets the per-frame result. A known player is
    accepted after ``votes_needed`` frames; for an unknown face one aligned crop
    is recorded every ``interval`` seconds, and after ``crops_needed`` crops it is
    a new player. Returns ``("known", name)``, ``("new", crops)`` or ``None``.
    Nothing happens while the face is outside the circle.

    One session belongs to one continuous person: if the face is out of the
    circle for longer than ``gap_tolerance`` seconds, collected crops and votes
    are discarded, so a different person cannot continue someone else's scan
    (#41, CR-03). Shorter gaps (a missed detection) are tolerated.
    """

    def __init__(self, crops_needed=config.FACE_SCAN_CROPS,
                 votes_needed=config.FACE_VOTES_NEEDED,
                 interval=config.FACE_SCAN_INTERVAL, time_fn=time.time,
                 gap_tolerance=config.FACE_SCAN_GAP):
        self.crops_needed = crops_needed
        self.votes_needed = votes_needed
        self.interval = interval
        self.gap_tolerance = gap_tolerance
        self._time = time_fn
        self.reset()

    def reset(self):
        self.votes = Counter()
        self.crops = []
        self._last_crop = None
        self._left_at = None             # when the face left the circle

    @property
    def progress(self):
        best_vote = max(self.votes.values(), default=0) / self.votes_needed
        return min(1.0, max(best_vote, len(self.crops) / self.crops_needed))

    def update(self, in_circle, name, crop):
        now = self._time()
        if not in_circle:
            if self._left_at is None:
                self._left_at = now
            elif now - self._left_at > self.gap_tolerance:
                self.reset()             # real interruption: start a new session
            return None
        if self._left_at is not None and now - self._left_at > self.gap_tolerance:
            self.reset()
        self._left_at = None
        if name is not None and name != GUEST:
            self.votes[name] += 1
            if self.votes[name] >= self.votes_needed:
                return "known", name
            return None
        if self._last_crop is None or now - self._last_crop >= self.interval:
            self.crops.append(crop)
            self._last_crop = now
        if len(self.crops) >= self.crops_needed:
            return "new", list(self.crops)
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
        self.model_paths = (detector_path, recognizer_path)
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

    def scan(self, frame):
        """``(name, similarity, box, crop)`` for the largest face, or all ``None``/0.

        The frame is downscaled for speed; ``box`` is (x, y, w, h) in frame
        coordinates, ``crop`` the aligned 112x112 face for enrolling.
        """
        scale = min(1.0, config.FACE_INPUT_WIDTH / frame.shape[1])
        small = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale,
                                                      interpolation=cv2.INTER_AREA)
        faces = self.detect(small)
        if not faces:
            return None, 0.0, None, None
        crop = self.aligned_crop(small, faces[0])
        name, score = best_match(self._feature_of_crop(crop), self.gallery, self.threshold)
        box = tuple(int(round(v / scale)) for v in faces[0][:4])
        return name, score, box, crop

    def largest_face_box(self, frame, width=320):
        """Box (x, y, w, h) of the largest face in frame coordinates, or None.

        Detection only (no recognition) on a copy downscaled to ``width`` -- cheap
        enough (~3 ms) to run every frame, e.g. to place the space suit.
        """
        scale = min(1.0, width / frame.shape[1])
        small = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale,
                                                      interpolation=cv2.INTER_AREA)
        faces = self.detect(small)
        if not faces:
            return None
        return tuple(int(round(v / scale)) for v in faces[0][:4])

    def identify(self, frame):
        """``(name, similarity, box)`` for the largest face, or ``(None, 0.0, None)``."""
        return self.scan(frame)[:3]

    def enroll(self, faces_dir, name, crops):
        """Save the crops to ``faces_dir/<name>/`` and add the player to the gallery."""
        folder = person_dir(faces_dir, name)
        os.makedirs(folder, exist_ok=True)
        stamp = int(time.time() * 1000)
        for i, crop in enumerate(crops):
            cv2.imwrite(os.path.join(folder, f"{stamp}_{i}.png"), crop)
        added = self.add_person(name.strip(), crops)
        logger.info("Enrolled player %s with %d face image(s)", name.strip(), added)
        return added


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
