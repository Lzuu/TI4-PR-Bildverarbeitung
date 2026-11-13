"""Verified model download shared by face_id, yolo_seg and pose_tracking (CR-15)."""

import hashlib
import io
import os
import urllib.request

import pytest

import config
import face_id
import model_download as md

PAYLOAD = b"model-bytes"
SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _serve(monkeypatch, data=PAYLOAD, seen=None):
    def fake_urlopen(url, timeout=None):
        if seen is not None:
            seen["timeout"] = timeout
        return io.BytesIO(data)
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)


def test_a_good_download_is_installed_and_leaves_no_part_file(monkeypatch, tmp_path):
    seen = {}
    _serve(monkeypatch, seen=seen)
    path = md.ensure_file(str(tmp_path / "m.onnx"), "http://example.invalid/m", SHA, timeout=7)
    assert open(path, "rb").read() == PAYLOAD
    assert os.listdir(tmp_path) == ["m.onnx"]
    assert seen["timeout"] == 7


def test_a_download_with_the_wrong_hash_leaves_no_file(monkeypatch, tmp_path):
    _serve(monkeypatch, data=b"truncated")
    with pytest.raises(RuntimeError, match="Hash stimmt nicht"):
        md.ensure_file(str(tmp_path / "m.onnx"), "http://example.invalid/m", SHA)
    assert os.listdir(tmp_path) == []


def test_a_stalled_download_fails_cleanly(monkeypatch, tmp_path):
    def stalled(url, timeout=None):
        raise TimeoutError("timed out")
    monkeypatch.setattr(urllib.request, "urlopen", stalled)
    with pytest.raises(RuntimeError, match="fehlgeschlagen"):
        md.ensure_file(str(tmp_path / "m.onnx"), "http://example.invalid/m", SHA)
    assert os.listdir(tmp_path) == []


def test_an_oversized_download_is_refused(monkeypatch, tmp_path):
    _serve(monkeypatch)
    with pytest.raises(RuntimeError, match="zu gross"):
        md.ensure_file(str(tmp_path / "m.onnx"), "http://example.invalid/m", SHA, max_bytes=5)
    assert os.listdir(tmp_path) == []


def test_a_corrupt_file_from_an_earlier_run_is_replaced(monkeypatch, tmp_path):
    (tmp_path / "m.onnx").write_bytes(b"half a model")
    _serve(monkeypatch)
    path = md.ensure_file(str(tmp_path / "m.onnx"), "http://example.invalid/m", SHA)
    assert open(path, "rb").read() == PAYLOAD


def test_a_corrupt_file_is_kept_and_reported_when_downloading_is_off(tmp_path):
    (tmp_path / "m.onnx").write_bytes(b"half a model")
    with pytest.raises(RuntimeError, match="Hash stimmt nicht"):
        md.ensure_file(str(tmp_path / "m.onnx"), "http://example.invalid/m", SHA, download=False)
    assert (tmp_path / "m.onnx").read_bytes() == b"half a model"


def test_a_missing_file_without_download_is_reported(tmp_path):
    with pytest.raises(RuntimeError, match="fehlt"):
        md.ensure_file(str(tmp_path / "m.onnx"), "http://example.invalid/m", SHA, download=False)


def test_the_face_models_are_downloaded_with_a_timeout(monkeypatch, tmp_path):
    seen = {}
    _serve(monkeypatch, data=b"x", seen=seen)
    with pytest.raises(RuntimeError):
        face_id.ensure_models(str(tmp_path))
    assert seen["timeout"] == config.MODEL_DOWNLOAD_TIMEOUT
    assert os.listdir(tmp_path) == []


def test_a_connection_dropped_mid_download_is_a_clean_failure(monkeypatch, tmp_path):
    import http.client

    def dropped(url, timeout=None):
        raise http.client.IncompleteRead(b"half")
    monkeypatch.setattr(urllib.request, "urlopen", dropped)
    with pytest.raises(RuntimeError, match="fehlgeschlagen"):
        md.ensure_file(str(tmp_path / "m.onnx"), "http://example.invalid/m", SHA)
    assert os.listdir(tmp_path) == []
