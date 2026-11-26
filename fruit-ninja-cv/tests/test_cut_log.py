"""Evaluation of the cut decisions logged during the practical test (#33)."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "analyze_cuts", Path(__file__).resolve().parent.parent / "tools" / "analyze_cuts.py")
analyze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analyze)

LOG = """\
2026-11-12 12:00:01 [INFO] game: Combo x3 -> +3 (score now 9)
2026-11-12 12:00:02 [INFO] challenge: Cut ok: direction=right speed=2.10 deviation=8
2026-11-12 12:00:03 [INFO] challenge: Cut ok: direction=up speed=1.40 deviation=21
2026-11-12 12:00:04 [INFO] challenge: Cut wrong_direction: direction=left speed=2.80 deviation=172
2026-11-12 12:00:05 [INFO] challenge: Cut too_slow: direction=down speed=0.45 deviation=12
2026-11-12 12:00:06 [WARNING] challenge: something else entirely
2026-11-12 12:00:07 [INFO] challenge: Cut ok: direction=right speed=3.00 deviation=4
"""


def test_parse_reads_only_cut_lines():
    records = analyze.parse_log(LOG.splitlines())
    assert [r["result"] for r in records] == ["ok", "ok", "wrong_direction", "too_slow", "ok"]
    assert records[0] == {"result": "ok", "direction": "right", "speed": 2.10, "deviation": 8.0}


def test_summary_counts_and_statistics():
    summary = analyze.summarize(analyze.parse_log(LOG.splitlines()))
    assert summary["ok"]["count"] == 3
    assert summary["ok"]["speed_min"] == pytest.approx(1.40)
    assert summary["ok"]["deviation_max"] == pytest.approx(21.0)
    assert summary["wrong_direction"]["count"] == 1
    assert summary["too_slow"]["speed_max"] == pytest.approx(0.45)


def test_summary_of_nothing_is_empty():
    assert analyze.summarize([]) == {}


def test_report_mentions_the_current_thresholds_and_the_margins():
    text = analyze.report(analyze.summarize(analyze.parse_log(LOG.splitlines())))
    assert "ok" in text and "too_slow" in text
    assert "CUT_MIN_SPEED" in text and "CUT_TOLERANCE_DEG" in text
    # slowest accepted cut vs fastest rejected slow touch: the margin is shown
    assert "1.40" in text and "0.45" in text


def test_report_without_data_says_so():
    assert "keine" in analyze.report({}).lower()
