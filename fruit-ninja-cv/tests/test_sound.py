"""Sound effects: files are written only when missing, Windows plays through winsound (#74).

No real sound is played: ``subprocess.Popen`` and ``winsound`` are replaced by fakes.
"""

import os
import subprocess

import pytest

import sound

NAMES = {"slice", "bomb", "miss", "combo", "over"}


class FakeWinsound:
    SND_FILENAME = 0x20000
    SND_ASYNC = 0x1

    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def PlaySound(self, path, flags):
        if self.fail:
            raise RuntimeError("no sound device")
        self.calls.append((path, flags))


def _files(directory):
    return {p[:-4] for p in os.listdir(directory) if p.endswith(".wav")}


# -- files are written only when they are missing -----------------------------------
def test_missing_sounds_are_created(tmp_path):
    sound.ensure_sounds(str(tmp_path))
    assert _files(tmp_path) == NAMES


def test_existing_sound_files_are_not_rewritten(tmp_path):
    sentinel = b"RIFF-already-there" + b"\0" * 100          # not a real wav, must stay untouched
    (tmp_path / "slice.wav").write_bytes(sentinel)
    sound.ensure_sounds(str(tmp_path))
    assert (tmp_path / "slice.wav").read_bytes() == sentinel
    assert _files(tmp_path) == NAMES                          # the missing ones were added


def test_nothing_is_synthesised_when_all_files_exist(tmp_path, monkeypatch):
    sound.ensure_sounds(str(tmp_path))

    def boom(*a, **k):
        raise AssertionError("synthesis must not run when every file exists")
    for fn in ("_swoosh", "_sine", "_noise"):
        monkeypatch.setattr(sound, fn, boom)
    sound.ensure_sounds(str(tmp_path))


def test_an_empty_or_truncated_file_counts_as_missing(tmp_path):
    (tmp_path / "bomb.wav").write_bytes(b"")                  # e.g. an interrupted write
    sound.ensure_sounds(str(tmp_path))
    assert (tmp_path / "bomb.wav").stat().st_size > 1000


def test_force_regenerates_everything(tmp_path):
    (tmp_path / "slice.wav").write_bytes(b"old" * 1000)
    sound.ensure_sounds(str(tmp_path), force=True)
    assert (tmp_path / "slice.wav").read_bytes()[:4] == b"RIFF"


def test_no_half_written_temp_files_are_left(tmp_path):
    sound.ensure_sounds(str(tmp_path))
    assert not [p for p in os.listdir(tmp_path) if p.endswith(".tmp")]


def test_player_does_not_rewrite_existing_files(tmp_path):
    sentinel = b"RIFF-keep-me" + b"\0" * 100
    for name in NAMES:
        (tmp_path / f"{name}.wav").write_bytes(sentinel)
    sound.SoundPlayer(str(tmp_path), platform="win32", winsound=FakeWinsound())
    sound.SoundPlayer(str(tmp_path), platform="darwin", afplay="afplay")
    assert all((tmp_path / f"{n}.wav").read_bytes() == sentinel for n in NAMES)


# -- Windows: winsound, no extra dependency ------------------------------------------
def test_sound_is_on_under_windows_when_winsound_is_available(tmp_path):
    player = sound.SoundPlayer(str(tmp_path), platform="win32", winsound=FakeWinsound())
    assert player.enabled


def test_windows_plays_the_file_asynchronously(tmp_path):
    fake = FakeWinsound()
    player = sound.SoundPlayer(str(tmp_path), platform="win32", winsound=fake)
    player.play("slice")
    path = os.path.join(str(tmp_path), "slice.wav")
    assert fake.calls == [(path, fake.SND_FILENAME | fake.SND_ASYNC)]


def test_unknown_sound_is_ignored(tmp_path):
    fake = FakeWinsound()
    sound.SoundPlayer(str(tmp_path), platform="win32", winsound=fake).play("nope")
    assert fake.calls == []


def test_a_failing_sound_device_does_not_crash_the_game(tmp_path, caplog):
    fake = FakeWinsound(fail=True)
    player = sound.SoundPlayer(str(tmp_path), platform="win32", winsound=fake)
    player.play("slice")
    player.play("slice")                                      # still alive, still quiet
    assert any("Failed to play sound" in r.getMessage() for r in caplog.records)


def test_windows_without_winsound_stays_quiet(tmp_path, monkeypatch):
    monkeypatch.setattr(sound, "_import_winsound", lambda: None)
    player = sound.SoundPlayer(str(tmp_path), platform="win32")
    assert not player.enabled
    player.play("slice")                                      # no error
    assert _files(tmp_path) == set()                          # and nothing is synthesised


# -- macOS: afplay as before -------------------------------------------------------------
def test_macos_plays_through_afplay(tmp_path, monkeypatch):
    started = []
    monkeypatch.setattr(subprocess, "Popen", lambda cmd, **k: started.append(cmd))
    player = sound.SoundPlayer(str(tmp_path), platform="darwin", afplay="/usr/bin/afplay")
    player.play("combo")
    assert started == [["/usr/bin/afplay", os.path.join(str(tmp_path), "combo.wav")]]


def test_other_systems_without_afplay_are_silent(tmp_path, monkeypatch):
    monkeypatch.setattr(sound.shutil, "which", lambda name: None)
    player = sound.SoundPlayer(str(tmp_path), platform="linux")
    assert not player.enabled
    player.play("slice")
    assert _files(tmp_path) == set()


def test_null_sound_accepts_any_name():
    sound.NullSound().play("anything")
