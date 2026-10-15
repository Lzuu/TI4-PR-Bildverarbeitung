"""Tests for the per-difficulty Top-3 highscore persistence."""

import highscore


def test_empty_when_file_missing(tmp_path):
    assert highscore.load(str(tmp_path / "hs.json"), "Mittel") == []


def test_keeps_top_three_per_difficulty(tmp_path):
    p = str(tmp_path / "hs.json")
    for s in [10, 50, 30, 20, 40]:
        highscore.add(p, "Schwer", s)
    assert highscore.load(p, "Schwer") == [50, 40, 30]


def test_difficulties_have_separate_boards(tmp_path):
    p = str(tmp_path / "hs.json")
    highscore.add(p, "Einfach", 100)
    highscore.add(p, "Schwer", 5)
    assert highscore.load(p, "Einfach") == [100]
    assert highscore.load(p, "Schwer") == [5]


def test_add_reports_rank_for_new_best(tmp_path):
    p = str(tmp_path / "hs.json")
    highscore.add(p, "Mittel", 10)
    scores, rank = highscore.add(p, "Mittel", 100)
    assert rank == 1 and scores[0] == 100


def test_low_score_not_ranked_once_full(tmp_path):
    p = str(tmp_path / "hs.json")
    for s in [100, 90, 80]:
        highscore.add(p, "Mittel", s)
    scores, rank = highscore.add(p, "Mittel", 5)
    assert rank is None and scores == [100, 90, 80]


def test_corrupt_file_is_treated_as_empty(tmp_path):
    f = tmp_path / "hs.json"
    f.write_text("this is not json")
    assert highscore.load(str(f), "Mittel") == []
