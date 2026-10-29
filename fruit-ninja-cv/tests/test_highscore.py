"""Tests for the per-difficulty Top-3 highscore persistence."""

import json

import highscore


def _scores(board):
    return [e["score"] for e in board]


def test_empty_when_file_missing(tmp_path):
    assert highscore.load(str(tmp_path / "hs.json"), "Mittel") == []


def test_keeps_top_three_per_difficulty(tmp_path):
    p = str(tmp_path / "hs.json")
    for s in [10, 50, 30, 20, 40]:
        highscore.add(p, "Schwer", s)
    assert _scores(highscore.load(p, "Schwer")) == [50, 40, 30]


def test_difficulties_have_separate_boards(tmp_path):
    p = str(tmp_path / "hs.json")
    highscore.add(p, "Einfach", 100)
    highscore.add(p, "Schwer", 5)
    assert _scores(highscore.load(p, "Einfach")) == [100]
    assert _scores(highscore.load(p, "Schwer")) == [5]


def test_add_reports_rank_for_new_best(tmp_path):
    p = str(tmp_path / "hs.json")
    highscore.add(p, "Mittel", 10)
    scores, rank = highscore.add(p, "Mittel", 100)
    assert rank == 1 and scores[0]["score"] == 100


def test_low_score_not_ranked_once_full(tmp_path):
    p = str(tmp_path / "hs.json")
    for s in [100, 90, 80]:
        highscore.add(p, "Mittel", s)
    scores, rank = highscore.add(p, "Mittel", 5)
    assert rank is None and _scores(scores) == [100, 90, 80]


def test_corrupt_file_is_treated_as_empty(tmp_path):
    f = tmp_path / "hs.json"
    f.write_text("this is not json")
    assert highscore.load(str(f), "Mittel") == []


# -- player names (issue #28) -------------------------------------------------
def test_entry_stores_the_player_name(tmp_path):
    p = str(tmp_path / "hs.json")
    highscore.add(p, "Mittel", 42, name="Noah Braun")
    assert highscore.load(p, "Mittel") == [{"name": "Noah Braun", "score": 42}]


def test_without_name_the_entry_belongs_to_a_guest(tmp_path):
    p = str(tmp_path / "hs.json")
    highscore.add(p, "Mittel", 7)
    assert highscore.load(p, "Mittel")[0]["name"] == highscore.DEFAULT_NAME == "Gast"


def test_rank_points_at_the_new_entry_on_equal_scores(tmp_path):
    p = str(tmp_path / "hs.json")
    highscore.add(p, "Mittel", 50, name="Leon")
    scores, rank = highscore.add(p, "Mittel", 50, name="Noah")
    assert scores[rank - 1]["name"] == "Noah"


def test_old_files_without_names_still_load(tmp_path):
    p = tmp_path / "hs.json"
    p.write_text(json.dumps({"Mittel": [30, 80]}))
    assert highscore.load(str(p), "Mittel") == [
        {"name": "Gast", "score": 80}, {"name": "Gast", "score": 30}]


# -- CR-07 (#41): read errors neither crash nor destroy the stored ranking ---
import builtins

import pytest


def _deny(monkeypatch, path, modes):
    """Make open() fail with PermissionError for ``path`` in the given modes."""
    real_open = builtins.open

    def guarded(file, mode="r", *a, **k):
        if str(file).startswith(str(path)) and any(m in mode for m in modes):
            raise PermissionError(f"locked: {file}")
        return real_open(file, mode, *a, **k)
    monkeypatch.setattr(builtins, "open", guarded)


def test_read_error_does_not_raise(tmp_path, monkeypatch):
    p = tmp_path / "hs.json"
    highscore.add(str(p), "Mittel", 50, name="Leon")
    _deny(monkeypatch, p, "r")
    assert highscore.load(str(p), "Mittel") == []                     # no exception
    result = highscore.add_entry(str(p), "Mittel", 10, name="Noah")
    assert not result.saved and result.reason == "unreadable"


def test_read_error_does_not_overwrite_the_existing_ranking(tmp_path, monkeypatch):
    p = tmp_path / "hs.json"
    highscore.add(str(p), "Mittel", 50, name="Leon")
    before = p.read_text(encoding="utf-8")
    _deny(monkeypatch, p, "r")
    highscore.add_entry(str(p), "Mittel", 99, name="Noah")
    monkeypatch.undo()
    assert p.read_text(encoding="utf-8") == before                    # old ranking intact


def test_write_error_is_reported(tmp_path, monkeypatch):
    p = tmp_path / "hs.json"
    _deny(monkeypatch, p, "w")
    result = highscore.add_entry(str(p), "Mittel", 10)
    assert not result.saved and result.reason == "write failed"
    assert result.rank == 1                                           # still shown this round


def test_successful_save_is_reported(tmp_path):
    result = highscore.add_entry(str(tmp_path / "hs.json"), "Mittel", 10, name="Leon")
    assert result.saved and result.reason is None and result.rank == 1


def test_corrupt_file_is_kept_as_backup_before_overwriting(tmp_path):
    p = tmp_path / "hs.json"
    p.write_text("this is not json")
    result = highscore.add_entry(str(p), "Mittel", 10)
    assert result.saved
    assert (tmp_path / "hs.json.corrupt").read_text() == "this is not json"
    assert highscore.load(str(p), "Mittel")[0]["score"] == 10
