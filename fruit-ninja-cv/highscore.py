"""Persistent Top-N highscores, kept separately per difficulty.

Stored as a single JSON object mapping a difficulty name to its sorted list of
entries, e.g. ``{"Einfach": [{"name": "Noah Braun", "score": 80}], "Schwer": []}``.
Old files with plain score lists (``{"Einfach": [80, 50]}``) are still read;
those scores belong to ``DEFAULT_NAME``.
"""

import json
import logging

logger = logging.getLogger(__name__)

TOP_N = 3
DEFAULT_NAME = "Gast"


def _entry(value):
    if isinstance(value, dict):
        return {"name": str(value.get("name", DEFAULT_NAME)), "score": int(value["score"])}
    return {"name": DEFAULT_NAME, "score": int(value)}


def _load_all(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: sorted((_entry(e) for e in v), key=lambda e: e["score"], reverse=True)
                    for k, v in data.items()}
    except (FileNotFoundError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        pass
    return {}


def load(path, key, top=TOP_N):
    """Return the Top-N entries for one difficulty (empty list on any error)."""
    return _load_all(path).get(key, [])[:top]


def add(path, key, score, name=DEFAULT_NAME, top=TOP_N):
    """Insert ``score`` of player ``name`` for difficulty ``key`` and persist.

    Returns ``(entries, rank)`` -- the new Top-N for that difficulty and the
    1-based rank of the new entry (or ``None`` if it did not make the list).
    """
    entry = {"name": name, "score": int(score)}
    all_scores = _load_all(path)
    board = all_scores.get(key, [])
    board.append(entry)
    board.sort(key=lambda e: e["score"], reverse=True)   # stable: older ties stay ahead
    board = board[:top]
    all_scores[key] = board

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_scores, f, ensure_ascii=False)
    except OSError:
        logger.warning("Could not write highscore file: %s", path)

    rank = next((i + 1 for i, e in enumerate(board) if e is entry), None)
    return board, rank
