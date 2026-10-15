"""Persistent Top-N highscores, kept separately per difficulty.

Stored as a single JSON object mapping a difficulty name to its sorted score
list, e.g. ``{"Einfach": [80, 50, 20], "Schwer": [30]}``.
"""

import json
import logging

logger = logging.getLogger(__name__)

TOP_N = 3


def _load_all(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {k: sorted((int(s) for s in v), reverse=True)
                    for k, v in data.items()}
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return {}


def load(path, key, top=TOP_N):
    """Return the Top-N scores for one difficulty (empty list on any error)."""
    return _load_all(path).get(key, [])[:top]


def add(path, key, score, top=TOP_N):
    """Insert ``score`` for difficulty ``key`` and persist.

    Returns ``(scores, rank)`` -- the new Top-N for that difficulty and the
    1-based rank of the new score (or ``None`` if it did not make the list).
    """
    score = int(score)
    all_scores = _load_all(path)
    board = all_scores.get(key, [])
    board.append(score)
    board.sort(reverse=True)
    board = board[:top]
    all_scores[key] = board

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(all_scores, f)
    except OSError:
        logger.warning("Could not write highscore file: %s", path)

    rank = board.index(score) + 1 if score in board else None
    return board, rank
