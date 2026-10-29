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
# Face scan in the circle (pure, injectable clock)
# --------------------------------------------------------------------------- #
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


CROP = np.zeros((112, 112, 3), np.uint8)


def _scanner(clock, crops=3, votes=3, interval=0.3, gap=None):
    kw = {} if gap is None else {"gap_tolerance": gap}
    return face_id.FaceScanner(crops_needed=crops, votes_needed=votes, interval=interval,
                               time_fn=clock, **kw)


def test_scan_recognises_known_player_after_enough_frames():
    clock = FakeClock()
    scan = _scanner(clock)
    assert scan.update(True, "Noah", CROP) is None
    assert scan.update(True, "Noah", CROP) is None
    assert scan.update(True, "Noah", CROP) == ("known", "Noah")


def test_scan_collects_crops_of_an_unknown_face():
    clock = FakeClock()
    scan = _scanner(clock, crops=3)
    results = []
    for i in range(3):
        clock.t = i * 0.5
        results.append(scan.update(True, face_id.GUEST, CROP))
    assert results[:2] == [None, None]
    kind, crops = results[2]
    assert kind == "new" and len(crops) == 3


def test_scan_takes_crops_only_every_interval():
    clock = FakeClock()
    scan = _scanner(clock, crops=3, interval=0.3)
    for _ in range(10):                       # same instant: only one crop
        scan.update(True, face_id.GUEST, CROP)
    assert len(scan.crops) == 1


def test_scan_pauses_while_face_is_outside_the_circle():
    clock = FakeClock()
    scan = _scanner(clock)
    for i in range(10):
        clock.t = i
        assert scan.update(False, "Noah", CROP) is None
    assert scan.progress == 0.0


def test_scan_progress_grows_towards_one():
    clock = FakeClock()
    scan = _scanner(clock, crops=4)
    scan.update(True, face_id.GUEST, CROP)
    clock.t = 1
    scan.update(True, face_id.GUEST, CROP)
    assert scan.progress == 0.5


def test_scan_reset_starts_over():
    clock = FakeClock()
    scan = _scanner(clock)
    scan.update(True, "Noah", CROP)
    scan.reset()
    assert scan.progress == 0.0 and scan.crops == []


def test_face_in_circle():
    circle = (500, 300, 150)
    assert face_id.face_in_circle((450, 250, 100, 110), circle)       # centred, fitting
    assert not face_id.face_in_circle((50, 50, 100, 110), circle)     # far outside
    assert not face_id.face_in_circle((480, 290, 20, 22), circle)     # too small (far away)
    assert not face_id.face_in_circle(None, circle)


def test_default_player_name_uses_next_free_number():
    assert face_id.default_player_name([]) == "Spieler 1"
    assert face_id.default_player_name(["Spieler 1", "Noah", "Spieler 3"]) == "Spieler 2"


# --------------------------------------------------------------------------- #
# Start flow: the face scan is the very first screen
# --------------------------------------------------------------------------- #
def test_game_starts_with_face_scan_when_identification_works():
    assert m.initial_state(object()) == m.STATE_FACE_SCAN


def test_game_starts_in_menu_without_identification():
    assert m.initial_state(None) == m.STATE_MENU


def test_menu_has_a_switch_player_target_apart_from_start_and_quit():
    import config
    w, h = config.WIDTH, config.HEIGHT
    sx, sy, sr = m.menu_start_target(w, h)
    for cx, cy, r in (m.menu_start_target(w, h), m.quit_target(w, h)):
        px, py, pr = m.player_target(w, h)
        assert (px - cx) ** 2 + (py - cy) ** 2 > (pr + r) ** 2


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


def test_largest_face_box_in_frame_coordinates(identifier):
    big = cv2.resize(_img("armstrong_portrait.jpg"), None, fx=3, fy=3)
    x, y, w, h = identifier.largest_face_box(big, width=320)
    face_small = identifier.detect(_img("armstrong_portrait.jpg"))[0]
    assert abs(x - 3 * face_small[0]) < 40 and abs(w - 3 * face_small[2]) < 40
    assert identifier.largest_face_box(np.zeros((480, 640, 3), np.uint8)) is None


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


def test_scan_returns_aligned_crop_of_largest_face(identifier):
    name, score, box, crop = identifier.scan(_img("armstrong_portrait.jpg"))
    assert box is not None
    assert crop.shape == (112, 112, 3)


def test_enroll_saves_crops_and_recognises_the_new_player(identifier, crew_faces, tmp_path):
    portrait = _img("armstrong_portrait.jpg")
    crop = identifier.aligned_crop(portrait, identifier.detect(portrait)[0])
    identifier.enroll(str(tmp_path), "Neil Armstrong", [crop, crop])
    assert len(list((tmp_path / "Neil Armstrong").glob("*.png"))) == 2
    img, (armstrong, _, _) = crew_faces
    assert identifier.match(img, armstrong)[0] == "Neil Armstrong"
    # a fresh identifier loads the saved crops from disk
    fresh = face_id.FaceIdentifier(*identifier.model_paths)
    assert fresh.load_gallery(str(tmp_path)) == 1


def test_ensure_models_rejects_corrupt_file(tmp_path):
    for model in face_id.MODELS.values():
        (tmp_path / model.filename).write_bytes(b"kaputt")
    with pytest.raises(RuntimeError):
        face_id.ensure_models(str(tmp_path), download=False)


# -- CR-03 (#41): one enrolment session = one continuous person --------------
def _crop(value):
    return np.full((112, 112, 3), value, np.uint8)


def test_scan_resets_after_a_real_interruption():
    clock = FakeClock()
    scan = _scanner(clock, crops=5, interval=0.3, gap=0.5)
    for i in range(2):
        clock.t = i * 0.4
        scan.update(True, face_id.GUEST, _crop(10))      # two crops of person A
    clock.t = 1.0
    scan.update(False, None, None)                       # left the circle ...
    clock.t = 2.0
    scan.update(False, None, None)                       # ... for a whole second
    clock.t = 2.1
    scan.update(True, face_id.GUEST, _crop(200))
    assert [c.mean() for c in scan.crops] == [200]       # old crops discarded


def test_person_a_leaves_and_b_enters_gives_no_mixed_profile():
    clock = FakeClock()
    scan = _scanner(clock, crops=3, interval=0.3, gap=0.5)
    for i in range(2):
        clock.t = i * 0.4
        scan.update(True, face_id.GUEST, _crop(10))      # A
    clock.t = 1.5
    scan.update(False, None, None)
    result = None
    for i in range(3):
        clock.t = 3.0 + i * 0.4
        result = scan.update(True, face_id.GUEST, _crop(200)) or result   # B
    kind, crops = result
    assert kind == "new" and [c.mean() for c in crops] == [200, 200, 200]


def test_known_votes_are_discarded_after_an_interruption():
    clock = FakeClock()
    scan = _scanner(clock, votes=3, gap=0.5)
    scan.update(True, "Noah", CROP)
    scan.update(True, "Noah", CROP)                      # 2 of 3 votes
    clock.t = 0.2
    scan.update(False, None, None)
    clock.t = 1.5
    assert scan.update(True, "Noah", CROP) is None       # starts again from 1 vote
    assert scan.votes["Noah"] == 1


def test_short_detection_dropout_is_tolerated():
    clock = FakeClock()
    scan = _scanner(clock, crops=5, interval=0.3, gap=0.5)
    clock.t = 0.0
    scan.update(True, face_id.GUEST, _crop(10))
    clock.t = 0.2
    scan.update(False, None, None)                       # one missed frame
    clock.t = 0.4
    scan.update(True, face_id.GUEST, _crop(10))
    assert len(scan.crops) == 2                          # still the same session
