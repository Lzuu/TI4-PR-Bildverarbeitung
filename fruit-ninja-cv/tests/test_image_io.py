"""Image files with non-ASCII paths (cv2.imread/imwrite fail on them under Windows)."""

import numpy as np

from utils import imread, imwrite


def test_image_roundtrip_in_a_folder_with_umlaut(tmp_path):
    folder = tmp_path / "Jürgen"
    folder.mkdir()
    img = np.full((8, 8, 3), 77, np.uint8)
    assert imwrite(str(folder / "a.png"), img)
    assert np.array_equal(imread(str(folder / "a.png")), img)


def test_missing_or_broken_files_give_none(tmp_path):
    assert imread(str(tmp_path / "fehlt.png")) is None
    (tmp_path / "kaputt.png").write_bytes(b"no image")
    assert imread(str(tmp_path / "kaputt.png")) is None
    assert not imwrite(str(tmp_path / "ordner" / "a.png"), np.zeros((2, 2, 3), np.uint8))
