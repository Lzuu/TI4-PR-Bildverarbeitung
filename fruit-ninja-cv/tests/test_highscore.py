"""Tests for the Top-3 highscore persistence."""

import highscore


def test_empty_when_file_missing(tmp_path):
    assert highscore.load(str(tmp_path / "hs.json")) == []


def test_keeps_only_top_three_sorted(tmp_path):
    p = str(tmp_path / "hs.json")
    for s in [10, 50, 30, 20, 40]:
        highscore.add(p, s)
    assert highscore.load(p) == [50, 40, 30]


def test_add_reports_rank_for_new_best(tmp_path):
    p = str(tmp_path / "hs.json")
    highscore.add(p, 10)
    scores, rank = highscore.add(p, 100)
    assert rank == 1
    assert scores[0] == 100


def test_low_score_is_not_ranked_once_list_is_full(tmp_path):
    p = str(tmp_path / "hs.json")
    for s in [100, 90, 80]:
        highscore.add(p, s)
    scores, rank = highscore.add(p, 5)
    assert rank is None
    assert scores == [100, 90, 80]


def test_corrupt_file_is_treated_as_empty(tmp_path):
    f = tmp_path / "hs.json"
    f.write_text("this is not json")
    assert highscore.load(str(f)) == []
