"""Sound effects.

Short effect sounds are *synthesised* with numpy on first run and written as WAV
files into ``assets/sounds/`` -- so the game needs no external audio assets. They
are played non-blocking via the macOS ``afplay`` command. On systems without
``afplay`` the player silently disables itself.
"""

import logging
import os
import shutil
import subprocess
import wave

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 44100
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


def _write_wav(path, samples):
    samples = np.clip(samples, -1.0, 1.0)
    data = (samples * 32767).astype("<i2").tobytes()
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(data)


def _build_sounds():
    """Return a dict name -> samples for every effect."""
    return {
        "slice": 0.7 * _chirp(700, 1500, 0.12, decay=10) + 0.2 * _noise(0.12, 20),
        "bomb": 0.9 * _sine(70, 0.5, decay=4) + 0.5 * _noise(0.5, 6),
        "miss": 0.6 * _sine(200, 0.18, decay=8),
        "combo": np.concatenate([_sine(523, 0.08, 6), _sine(659, 0.08, 6),
                                 _sine(784, 0.12, 6)]),
        "over": np.concatenate([_sine(400, 0.14, 5), _sine(300, 0.14, 5),
                                _sine(200, 0.22, 4)]),
    }


def ensure_sounds(sound_dir=SOUND_DIR):
    """Generate any missing WAV files. Returns the directory."""
    os.makedirs(sound_dir, exist_ok=True)
    sounds = _build_sounds()
    for name, samples in sounds.items():
        path = os.path.join(sound_dir, name + ".wav")
        if not os.path.exists(path):
            _write_wav(path, samples)
    return sound_dir


# --------------------------------------------------------------------------- #
# Players
# --------------------------------------------------------------------------- #
class NullSound:
    """No-op player used as the default (e.g. in tests)."""

    def play(self, name):
        pass


class SoundPlayer:
    """Plays synthesised effects non-blocking via ``afplay`` (macOS)."""

    def __init__(self, sound_dir=SOUND_DIR):
        self.dir = sound_dir
        self._afplay = shutil.which("afplay")
        self.enabled = self._afplay is not None
        if self.enabled:
            try:
                ensure_sounds(sound_dir)
            except Exception:
                logger.exception("Could not generate sound files -- disabling sound")
                self.enabled = False
        else:
            logger.warning("afplay not found -- sound disabled")

    def play(self, name):
        if not self.enabled:
            return
        path = os.path.join(self.dir, name + ".wav")
        if not os.path.exists(path):
            return
        try:
            subprocess.Popen([self._afplay, path],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            logger.exception("Failed to play sound %s", name)
