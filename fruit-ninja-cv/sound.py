"""Sound effects.

Short effect sounds are *synthesised* with numpy and written as WAV files into
``assets/sounds/`` -- so the game needs no external audio assets. Files that already
exist are kept (to change a sound, delete its file or call ``ensure_sounds(force=True)``).
They are played non-blocking: on Windows with ``winsound.PlaySound`` (standard library,
``SND_ASYNC``: a new effect cuts the previous one off), on macOS via the ``afplay``
command. Anywhere else the player silently disables itself.
"""

import logging
import os
import shutil
import subprocess
import sys
import wave

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
WAV_HEADER_BYTES = 44             # a file this small cannot hold a sound (interrupted write)
SOUND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "sounds")


# --------------------------------------------------------------------------- #
# Synthesis
# --------------------------------------------------------------------------- #
def _envelope(n, decay):
    return np.exp(-decay * np.linspace(0, 1, n))


def _sine(freq, dur, decay=5.0):
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    return np.sin(2 * np.pi * freq * t) * _envelope(len(t), decay)


def _chirp(f0, f1, dur, decay=6.0):
    t = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    freq = np.linspace(f0, f1, len(t))
    phase = 2 * np.pi * np.cumsum(freq) / SAMPLE_RATE
    return np.sin(phase) * _envelope(len(t), decay)


def _noise(dur, decay=8.0):
    n = int(SAMPLE_RATE * dur)
    return np.random.uniform(-1, 1, n) * _envelope(n, decay)


def _swoosh(dur=0.16):
    """A blade-slash: softened noise with a fast attack + a descending whoosh."""
    n = int(SAMPLE_RATE * dur)
    # Low-pass the noise a little (moving average) so it is airy, not harsh
    noise = np.random.uniform(-1, 1, n)
    k = 12
    noise = np.convolve(noise, np.ones(k) / k, mode="same")
    t = np.linspace(0, 1, n)
    attack = np.clip(t / 0.04, 0, 1)         # ~4% quick attack
    decay = np.exp(-6.0 * t)                  # fast decay
    env = attack * decay
    # A descending tone gives the "slash" its pitch movement
    freq = np.linspace(1700, 260, n)
    tone = 0.25 * np.sin(2 * np.pi * np.cumsum(freq) / SAMPLE_RATE) * np.exp(-9.0 * t)
    return 0.9 * noise * env + tone


def _write_wav(path, samples):
    """Write through a temporary file, so an interrupted write leaves no broken sound."""
    samples = np.clip(samples, -1.0, 1.0)
    data = (samples * 32767).astype("<i2").tobytes()
    tmp = path + ".tmp"
    with wave.open(tmp, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data)
    os.replace(tmp, path)


# name -> function that synthesises it (called only for files that are missing)
BUILDERS = {
    "slice": lambda: _swoosh(0.16),
    "bomb": lambda: 0.9 * _sine(70, 0.5, decay=4) + 0.5 * _noise(0.5, 6),
    "miss": lambda: 0.6 * _sine(200, 0.18, decay=8),
    "combo": lambda: np.concatenate([_sine(523, 0.08, 6), _sine(659, 0.08, 6),
                                     _sine(784, 0.12, 6)]),
    "over": lambda: np.concatenate([_sine(400, 0.14, 5), _sine(300, 0.14, 5),
                                    _sine(200, 0.22, 4)]),
}


def _is_missing(path):
    try:
        return os.path.getsize(path) <= WAV_HEADER_BYTES
    except OSError:
        return True


def ensure_sounds(sound_dir=SOUND_DIR, force=False):
    """Write the WAV files that are missing (or all of them with ``force``). Returns dir."""
    os.makedirs(sound_dir, exist_ok=True)
    for name, build in BUILDERS.items():
        path = os.path.join(sound_dir, name + ".wav")
        if force or _is_missing(path):
            _write_wav(path, build())
    return sound_dir


# --------------------------------------------------------------------------- #
# Players
# --------------------------------------------------------------------------- #
class NullSound:
    """No-op player used as the default (e.g. in tests)."""

    def play(self, name):
        pass


def _import_winsound():
    try:
        import winsound
    except ImportError:
        return None
    return winsound


class SoundPlayer:
    """Plays the synthesised effects without blocking the game.

    ``platform``, ``winsound`` and ``afplay`` exist so tests can replace the real sound
    backends; the game just calls ``SoundPlayer()``.
    """

    def __init__(self, sound_dir=SOUND_DIR, platform=None, winsound=None, afplay=None):
        self.dir = sound_dir
        self._play_file = None
        platform = sys.platform if platform is None else platform
        if platform == "win32":
            winsound = winsound or _import_winsound()
            if winsound is not None:
                flags = winsound.SND_FILENAME | winsound.SND_ASYNC
                self._play_file = lambda path: winsound.PlaySound(path, flags)
            else:
                logger.warning("winsound not available -- sound disabled")
        else:
            afplay = afplay or shutil.which("afplay")
            if afplay is not None:
                self._play_file = lambda path: subprocess.Popen(
                    [afplay, path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                logger.warning("afplay not found -- sound disabled")
        self.enabled = self._play_file is not None
        if self.enabled:
            try:
                ensure_sounds(sound_dir)
            except Exception:
                logger.exception("Could not generate sound files -- disabling sound")
                self.enabled = False

    def play(self, name):
        if not self.enabled:
            return
        path = os.path.join(self.dir, name + ".wav")
        if not os.path.exists(path):
            return
        try:
            self._play_file(path)
        except Exception:
            logger.exception("Failed to play sound %s", name)
