"""Persistent Top-N highscores, kept separately per difficulty.

Stored as a single JSON object mapping a difficulty name to its sorted list of
entries, e.g. ``{"Einfach": [{"name": "Noah Braun", "score": 80}], "Schwer": []}``.
Old files with plain score lists (``{"Einfach": [80, 50]}``) are still read;
those scores belong to ``DEFAULT_NAME``.

Errors are told apart (#41, CR-07): a missing file is simply empty; a corrupt
file is kept as ``<file>.corrupt`` before it is replaced; a file that cannot
be *read* (e.g. locked) is never overwritten, so a valid ranking is not lost;
a failed write is reported. Writes go through a temporary file, so an
interrupted write cannot leave half a file behind.
"""

import json
import logging
import os
from collections import namedtuple

logger = logging.getLogger(__name__)

TOP_N = 3
DEFAULT_NAME = "Gast"


def _entry(value):
    if isinstance(value, dict):
        return {"name": str(value.get("name", DEFAULT_NAME)), "score": int(value["score"])}
    return {"name": DEFAULT_NAME, "score": int(value)}


Result = namedtuple("Result", "board rank saved reason")


def _read(path):
    """``(all_scores, status)``; status is "ok", "missing", "corrupt" or "unreadable"."""
    try:
        with open(path, encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError:
        return {}, "missing"
    except OSError:
        logger.warning("Could not read highscore file: %s", path, exc_info=True)
        return {}, "unreadable"
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("highscore file is not a JSON object")
        return ({k: sorted((_entry(e) for e in v), key=lambda e: e["score"], reverse=True)
                 for k, v in data.items()}, "ok")
    except (ValueError, TypeError, KeyError):
        logger.warning("Highscore file %s is corrupt -- starting a new ranking", path)
        return {}, "corrupt"


def _write(path, all_scores):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(all_scores, f, ensure_ascii=False)
    os.replace(tmp, path)


def load(path, key, top=TOP_N):
    """Return the Top-N entries for one difficulty (empty list on any error)."""
    return _read(path)[0].get(key, [])[:top]


def add_entry(path, key, score, name=DEFAULT_NAME, top=TOP_N):
    """Insert ``score`` of player ``name`` for difficulty ``key`` and persist.

    Returns ``Result(board, rank, saved, reason)``: the Top-N to show, the
    1-based rank of the new entry (or None), whether it was saved and why not
    ("unreadable": existing file left untouched, "write failed").
    """
    entry = {"name": name, "score": int(score)}
    all_scores, status = _read(path)
    board = all_scores.get(key, [])
    board.append(entry)
    board.sort(key=lambda e: e["score"], reverse=True)   # stable: older ties stay ahead
    board = board[:top]
    all_scores[key] = board
    rank = next((i + 1 for i, e in enumerate(board) if e is entry), None)

    if status == "unreadable":
        logger.warning("Highscore not saved: %s could not be read (left unchanged)", path)
        return Result(board, rank, False, "unreadable")
    try:
        if status == "corrupt":
            with open(path, encoding="utf-8", errors="replace") as f, \
                    open(path + ".corrupt", "w", encoding="utf-8") as backup:
                backup.write(f.read())
        _write(path, all_scores)
    except OSError:
        logger.warning("Could not write highscore file: %s", path, exc_info=True)
        return Result(board, rank, False, "write failed")
    return Result(board, rank, True, None)


def add(path, key, score, name=DEFAULT_NAME, top=TOP_N):
    """Like ``add_entry`` but returns only ``(entries, rank)``."""
    result = add_entry(path, key, score, name, top)
    return result.board, result.rank
