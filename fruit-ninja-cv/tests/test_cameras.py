"""Tests for camera discovery, selection and persistence (issue #31).

All tests use fake captures, so they run without any real camera.
"""

import json

import cv2
import numpy as np

import cameras
import config
import main as m


class FakeCap:
    def __init__(self, frame=None, opens=True):
        self._frame = frame
        self._opens = opens
        self.released = False
        self.props = {}

    def isOpened(self):
        return self._opens and not self.released

    def read(self):
        if self._frame is None or self.released:
            return False, None
        return True, self._frame.copy()

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.props.get(prop, 0)

    def release(self):
        self.released = True


class FakeOpener:
    """Simulates the cameras of a machine: ``{index: frame or None}``."""

    def __init__(self, cams):
        self.cams = cams
        self.opened = []

    def __call__(self, index):
        if index not in self.cams:
            cap = FakeCap(opens=False)
        else:
            cap = FakeCap(self.cams[index])
        self.opened.append(cap)
        return cap


def frame(value=100):
    return np.full((1080, 1920, 3), value, np.uint8)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def test_probe_finds_only_working_cameras_and_releases_them():
    opener = FakeOpener({0: frame(), 2: frame()})
    found = cameras.probe(range(4), opener)
    assert sorted(found) == [0, 2]
    assert all(cap.released for cap in opener.opened)


def test_probe_skips_camera_without_picture():
    found = cameras.probe(range(2), FakeOpener({0: None, 1: frame()}))
    assert sorted(found) == [1]


def test_probe_returns_small_previews():
    preview = cameras.probe([0], FakeOpener({0: frame()}))[0]
    assert preview.shape[:2] == (cameras.PREVIEW_SIZE[1], cameras.PREVIEW_SIZE[0])


def test_choose_saved_camera_if_it_was_found():
    assert cameras.choose_index(2, 1, [0, 1, 2]) == 2


def test_choose_default_when_saved_camera_is_gone():
    assert cameras.choose_index(3, 1, [0, 1]) == 1


def test_choose_any_found_camera_as_last_resort():
    assert cameras.choose_index(3, 4, [0, 2]) == 0


def test_choose_nothing_without_cameras():
    assert cameras.choose_index(1, 1, []) is None


# --------------------------------------------------------------------------- #
# Opening and switching
# --------------------------------------------------------------------------- #
def test_candidate_order_saved_then_default_then_rest():
    assert cameras.candidate_order(2, 1, 4) == [2, 1, 0, 3]
    assert cameras.candidate_order(None, 1, 3) == [1, 0, 2]
    assert cameras.candidate_order(1, 1, 3) == [1, 0, 2]


def test_open_uses_saved_camera_if_available():
    cam = cameras.Camera(FakeOpener({0: frame(), 2: frame()}))
    assert cam.open_first(cameras.candidate_order(2, 1, 4))
    assert cam.index == 2


def test_open_falls_back_when_saved_camera_is_gone():
    cam = cameras.Camera(FakeOpener({0: frame()}))
    assert cam.open_first(cameras.candidate_order(2, 1, 4))
    assert cam.index == 0


def test_open_fails_without_any_camera():
    cam = cameras.Camera(FakeOpener({}))
    assert not cam.open_first(cameras.candidate_order(None, 1, 3))
    assert cam.index is None


def test_open_requests_full_hd_mjpg():
    opener = FakeOpener({0: frame()})
    cameras.Camera(opener).open_first([0])
    props = opener.opened[0].props
    assert props[cv2.CAP_PROP_FRAME_WIDTH] == 1920
    assert props[cv2.CAP_PROP_FRAME_HEIGHT] == 1080
    assert props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")


def test_switch_releases_old_camera():
    opener = FakeOpener({0: frame(10), 2: frame(20)})
    cam = cameras.Camera(opener)
    cam.open_first([0])
    old = cam.cap
    assert cam.switch(2)
    assert cam.index == 2 and old.released
    assert cam.read()[1][0, 0, 0] == 20


def test_switch_to_missing_camera_keeps_the_current_one():
    cam = cameras.Camera(FakeOpener({0: frame(10)}))
    cam.open_first([0])
    assert not cam.switch(3)
    assert cam.index == 0 and cam.read()[0]


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_setting_roundtrip(tmp_path):
    path = str(tmp_path / "settings.json")
    cameras.save_camera_index(path, 2)
    assert cameras.load_camera_index(path) == 2


def test_missing_or_corrupt_settings_give_none(tmp_path):
    assert cameras.load_camera_index(str(tmp_path / "fehlt.json")) is None
    bad = tmp_path / "settings.json"
    bad.write_text("kein json")
    assert cameras.load_camera_index(str(bad)) is None


def test_saving_keeps_other_settings(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"anderes": True}))
    cameras.save_camera_index(str(path), 1)
    assert json.loads(path.read_text()) == {"anderes": True, "camera_index": 1}


# --------------------------------------------------------------------------- #
# Menu layout
# --------------------------------------------------------------------------- #
def _overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def test_camera_button_only_with_more_than_one_camera():
    assert not m.show_camera_button([1])
    assert m.show_camera_button([0, 1])


def test_camera_button_does_not_overlap_pen_colour_button():
    w, h = config.WIDTH, config.HEIGHT
    assert not _overlap(m.camera_button(w, h), m.settings_button(w, h))


def _two_cameras():
    cam = cameras.Camera(FakeOpener({0: frame(10), 2: frame(20)}))
    cam.open_first([0])
    return cam


def test_click_on_camera_button_opens_the_panel():
    w, h = config.WIDTH, config.HEIGHT
    bx, by, bw, bh = m.camera_button(w, h)
    is_open = m.handle_camera_click((bx + 5, by + 5), False, [0, 2], _two_cameras(), w, h)
    assert is_open


def test_click_on_a_tile_switches_and_closes_the_panel(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SETTINGS_FILE", str(tmp_path / "settings.json"))
    w, h = config.WIDTH, config.HEIGHT
    cam = _two_cameras()
    x, y, tw, th = m.camera_tiles(w, h, 2)[1]               # tile of camera 2
    is_open = m.handle_camera_click((x + 5, y + 5), True, [0, 2], cam, w, h)
    assert not is_open and cam.index == 2
    assert cameras.load_camera_index(config.SETTINGS_FILE) == 2


def test_no_camera_button_with_a_single_camera():
    w, h = config.WIDTH, config.HEIGHT
    bx, by, bw, bh = m.camera_button(w, h)
    assert not m.handle_camera_click((bx + 5, by + 5), False, [0], _two_cameras(), w, h)


def test_space_background_reference_is_dropped_after_camera_switch():
    # the empty-background reference belongs to the camera it was taken with
    assert m.reference_still_valid(reference_camera=0, camera_index=0)
    assert not m.reference_still_valid(reference_camera=0, camera_index=2)


def test_guest_circle_on_face_scan_does_not_overlap_camera_button():
    w, h = config.WIDTH, config.HEIGHT
    cx, cy, r = m.guest_target(w, h)
    assert not _overlap((cx - r, cy - r, 2 * r, 2 * r), m.camera_button(w, h))


def test_camera_tiles_fit_on_screen_without_overlap():
    w, h = config.WIDTH, config.HEIGHT
    for n in range(2, 6):
        tiles = m.camera_tiles(w, h, n)
        assert len(tiles) == n
        for i, (x, y, tw, th) in enumerate(tiles):
            assert x >= 0 and y >= 0 and x + tw <= w and y + th <= h
            assert all(not _overlap(tiles[i], t) for t in tiles[i + 1:])
