"""Settings, highscores and logs live in the program folder, wherever the game is started (CR-16)."""

import os

import config
import logging_config

PROGRAM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_files_do_not_depend_on_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for path in (config.SETTINGS_FILE, config.HIGHSCORE_FILE, config.PERF_CSV,
                 logging_config.LOG_DIR):
        assert os.path.isabs(path), path
        assert os.path.commonpath([path, PROGRAM_DIR]) == PROGRAM_DIR, path
