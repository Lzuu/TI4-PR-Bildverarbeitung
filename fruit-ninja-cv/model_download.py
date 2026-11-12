"""Verified download of the ONNX models, shared by face_id, yolo_seg and pose_tracking (#67).

One code path for all three, so every model gets the same safeguards:

  * a timeout, so a stalled network fails after a while instead of hanging the start,
  * the file is hashed while it is written to ``<name>.part`` and only moved to its
    final name when the SHA256 matches -- a truncated or wrong download never
    becomes ``models/<name>``,
  * a corrupt file left behind by an earlier run is replaced by a fresh verified
    download (when downloading is allowed) instead of blocking the feature for good.

Every failure is a ``RuntimeError``; the callers switch the feature off with one log entry.
"""

import hashlib
import logging
import os
import urllib.request

import config

logger = logging.getLogger(__name__)

CHUNK = 1 << 20


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url, path, sha256, timeout, max_bytes=None):
    """Download ``url`` to ``path``; the file only appears there if its SHA256 matches."""
    name = os.path.basename(path)
    part = path + ".part"
    try:
        digest, size = hashlib.sha256(), 0
        with urllib.request.urlopen(url, timeout=timeout) as response, open(part, "wb") as f:
            for chunk in iter(lambda: response.read(CHUNK), b""):
                size += len(chunk)
                if max_bytes is not None and size > max_bytes:
                    raise RuntimeError(f"Download von {name} ist zu gross (mehr als {max_bytes} Bytes)")
                digest.update(chunk)
                f.write(chunk)
        if digest.hexdigest() != sha256:
            raise RuntimeError(f"Download von {name} ist beschaedigt (Hash stimmt nicht)")
        os.replace(part, path)
    except OSError as exc:                     # incl. TimeoutError and URLError
        raise RuntimeError(f"Download von {name} fehlgeschlagen: {exc}") from exc
    finally:
        if os.path.exists(part):
            os.remove(part)


def ensure_file(path, url, sha256, download=True, timeout=None, max_bytes=None, missing_hint=""):
    """Return ``path`` once the file exists with the right hash, downloading it if allowed.

    Raises ``RuntimeError`` if it is missing or corrupt and may not be downloaded, or if
    the download fails. Without ``download`` a corrupt file is left untouched.
    """
    name = os.path.basename(path)
    if os.path.exists(path):
        if sha256_of(path) == sha256:
            return path
        if not download:
            raise RuntimeError(f"{name} ist beschaedigt (Hash stimmt nicht)")
        logger.warning("%s ist beschaedigt -- wird neu geladen", name)
    elif not download:
        raise RuntimeError(f"{name} fehlt in {os.path.dirname(path)}{missing_hint}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info("Downloading %s ...", name)
    download_verified(url, path, sha256,
                      config.MODEL_DOWNLOAD_TIMEOUT if timeout is None else timeout, max_bytes)
    return path
