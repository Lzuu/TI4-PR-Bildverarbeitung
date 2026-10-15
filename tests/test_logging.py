"""Anforderung: Logfiles zum Debuggen."""
import logging

from log_setup import remove_old_logs, setup_logging


def test_logdatei_wird_angelegt_und_beschrieben(tmp_path):
    path = setup_logging("testprogramm", log_dir=str(tmp_path))
    logging.getLogger("test").debug("Debug-Meldung fuer die Datei")
    for handler in logging.getLogger().handlers:
        handler.flush()
    text = open(path, encoding="utf-8").read()
    assert "Debug-Meldung fuer die Datei" in text
    assert "DEBUG" in text


def test_alte_logdateien_werden_aufgeraeumt(tmp_path):
    for i in range(25):
        (tmp_path / f"prog_20260101_0000{i:02d}.log").write_text("x")
    (tmp_path / "anderes_20260101_000000.log").write_text("x")
    remove_old_logs(str(tmp_path), "prog", keep=20)
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert len([n for n in remaining if n.startswith("prog_")]) == 20
    assert "prog_20260101_000024.log" in remaining      # die neuesten bleiben
    assert "anderes_20260101_000000.log" in remaining   # fremde Logs bleiben unberuehrt
