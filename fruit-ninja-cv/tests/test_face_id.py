"""Tests for player identification by face (issue #28).

The pure logic (matching, voting, names) runs without models. The tests that
need the YuNet/SFace models download them on first use (cached in models/) and
use public-domain NASA photos from tests/data (see tests/data/QUELLEN.md).
"""

import os

import cv2
import numpy as np
import pytest

import face_id
import main as m

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _img(name):
    return cv2.imread(os.path.join(DATA, name))


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


# --------------------------------------------------------------------------- #
# Matching against the gallery (pure)
# --------------------------------------------------------------------------- #
def test_best_match_known_person():
    gallery = {"Noah": [_unit([1, 0, 0])], "Leon": [_unit([0, 1, 0])]}
    name, score = face_id.best_match(_unit([0.9, 0.1, 0]), gallery, threshold=0.5)
    assert name == "Noah"
    assert score > 0.9


def test_best_match_unknown_is_guest():
    gallery = {"Noah": [_unit([1, 0, 0])]}
    name, _ = face_id.best_match(_unit([0, 0, 1]), gallery, threshold=0.5)
    assert name == face_id.GUEST


def test_best_match_empty_gallery_is_guest():
    assert face_id.best_match(_unit([1, 0, 0]), {}, threshold=0.5)[0] == face_id.GUEST


def test_best_match_uses_best_reference_of_a_person():
    gallery = {"Noah": [_unit([0, 0, 1]), _unit([1, 0, 0])]}
    assert face_id.best_match(_unit([1, 0, 0]), gallery, threshold=0.5)[0] == "Noah"


# --------------------------------------------------------------------------- #
# Deciding over several frames (pure, injectable clock)
# --------------------------------------------------------------------------- #
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_vote_decides_after_enough_matching_frames():
    clock = FakeClock()
    vote = face_id.PlayerVote(needed=3, timeout=5.0, time_fn=clock)
    assert vote.update("Noah") is None
    assert vote.update("Noah") is None
    assert vote.update("Noah") == "Noah"


def test_vote_ignores_frames_without_face():
    vote = face_id.PlayerVote(needed=2, timeout=5.0, time_fn=FakeClock())
    assert vote.update(None) is None
    assert vote.update("Noah") is None
    assert vote.update(None) is None
    assert vote.update("Noah") == "Noah"


def test_vote_timeout_without_face_is_guest():
    clock = FakeClock()
    vote = face_id.PlayerVote(needed=3, timeout=4.0, time_fn=clock)
    vote.update(None)
    clock.t = 4.1
    assert vote.update(None) == face_id.GUEST


def test_vote_timeout_takes_most_frequent_name():
    clock = FakeClock()
    vote = face_id.PlayerVote(needed=10, timeout=4.0, time_fn=clock)
    for name in ["Noah", face_id.GUEST, "Noah"]:
        vote.update(name)
    clock.t = 4.1
    assert vote.update(None) == "Noah"


def test_vote_reset_starts_over():
    clock = FakeClock()
    vote = face_id.PlayerVote(needed=2, timeout=4.0, time_fn=clock)
    vote.update("Noah")
    clock.t = 10.0
    vote.reset()
    assert vote.update("Leon") is None      # old vote and old start time are gone


# --------------------------------------------------------------------------- #
# Names and enrollment folders (pure)
# --------------------------------------------------------------------------- #
def test_display_name_replaces_umlauts():
    assert face_id.display_name("Jürgen Größ") == "Juergen Groess"


@pytest.mark.parametrize("bad", ["", "   ", "../hack", "a/b", "a\\b", ".."])
def test_person_dir_rejects_unsafe_names(tmp_path, bad):
    with pytest.raises(ValueError):
        face_id.person_dir(str(tmp_path), bad)


def test_person_dir_inside_faces_dir(tmp_path):
    path = face_id.person_dir(str(tmp_path), " Noah Braun ")
    assert path == os.path.join(str(tmp_path), "Noah Braun")


# --------------------------------------------------------------------------- #
# New state: identification runs before the difficulty selection
# --------------------------------------------------------------------------- #
class _Ident:
    def __init__(self, persons):
        self.persons = persons


def test_start_goes_to_identification_when_players_are_enrolled():
    assert m.state_after_start(_Ident(["Noah"])) == m.STATE_IDENTIFY


def test_start_skips_identification_without_enrolled_players():
    assert m.state_after_start(_Ident([])) == m.STATE_DIFFICULTY
    assert m.state_after_start(None) == m.STATE_DIFFICULTY


# --------------------------------------------------------------------------- #
# Real models on real photos
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def model_paths():
    try:
        return face_id.ensure_models()
    except RuntimeError as exc:
        pytest.skip(f"Gesichtsmodelle nicht verfuegbar: {exc}")


@pytest.fixture
def identifier(model_paths):
    return face_id.FaceIdentifier(model_paths["detector"], model_paths["recognizer"])


@pytest.fixture
def crew_faces(identifier):
    """Faces of the Apollo 11 crew photo, left to right: Armstrong, Collins, Aldrin."""
    img = _img("apollo11_crew.jpg")
    faces = sorted(identifier.detect(img), key=lambda f: f[0])
    return img, faces


def test_detects_the_face_in_a_portrait(identifier):
    assert len(identifier.detect(_img("armstrong_portrait.jpg"))) == 1


def test_detects_all_three_faces_in_crew_photo(crew_faces):
    assert len(crew_faces[1]) == 3


def test_no_face_in_empty_image(identifier):
    assert identifier.detect(np.full((480, 640, 3), 128, np.uint8)) == []


def test_known_faces_are_recognised_in_a_different_photo(identifier, crew_faces):
    identifier.add_person("Neil Armstrong", [_img("armstrong_portrait.jpg")])
    identifier.add_person("Buzz Aldrin", [_img("aldrin_portrait.jpg")])
    img, (armstrong, _, aldrin) = crew_faces
    assert identifier.match(img, armstrong)[0] == "Neil Armstrong"
    assert identifier.match(img, aldrin)[0] == "Buzz Aldrin"


def test_unknown_face_is_guest(identifier, crew_faces):
    identifier.add_person("Neil Armstrong", [_img("armstrong_portrait.jpg")])
    img, (_, collins, _) = crew_faces
    assert identifier.match(img, collins)[0] == face_id.GUEST


def test_identify_uses_largest_face_and_returns_box_in_frame_coordinates(identifier):
    identifier.add_person("Neil Armstrong", [_img("armstrong_portrait.jpg")])
    big = cv2.resize(_img("armstrong_portrait.jpg"), None, fx=3, fy=3)  # larger than detector input
    name, score, box = identifier.identify(big)
    assert name == "Neil Armstrong"
    x, y, w, h = box
    face_small = identifier.detect(_img("armstrong_portrait.jpg"))[0]
    assert abs(x - 3 * face_small[0]) < 30 and abs(w - 3 * face_small[2]) < 30


def test_identify_without_face(identifier):
    assert identifier.identify(np.zeros((480, 640, 3), np.uint8)) == (None, 0.0, None)


def test_gallery_loads_photos_and_aligned_crops_from_folders(identifier, tmp_path):
    neil = tmp_path / "Neil Armstrong"
    neil.mkdir()
    portrait = _img("armstrong_portrait.jpg")
    cv2.imwrite(str(neil / "foto.jpg"), portrait)
    crop = identifier.aligned_crop(portrait, identifier.detect(portrait)[0])
    cv2.imwrite(str(neil / "crop.png"), crop)
    (tmp_path / "Leer").mkdir()            # person without usable images is skipped
    assert identifier.load_gallery(str(tmp_path)) == 1
    assert identifier.persons == ["Neil Armstrong"]
    assert len(identifier.gallery["Neil Armstrong"]) == 2


def test_ensure_models_rejects_corrupt_file(tmp_path):
    for model in face_id.MODELS.values():
        (tmp_path / model.filename).write_bytes(b"kaputt")
    with pytest.raises(RuntimeError):
        face_id.ensure_models(str(tmp_path), download=False)
