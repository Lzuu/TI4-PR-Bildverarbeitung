"""Persistent Top-N highscore list stored as JSON."""

import json
import logging

logger = logging.getLogger(__name__)

TOP_N = 3


def load(path, top=TOP_N):
    """Return the stored scores, sorted high to low (empty list on any error)."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        scores = sorted((int(s) for s in data), reverse=True)
        return scores[:top]
    except (FileNotFoundError, ValueError, TypeError, json.JSONDecodeError):
        return []


def add(path, score, top=TOP_N):
    """Insert ``score`` and persist the Top-N.

    Returns ``(scores, rank)`` where ``scores`` is the new Top-N and ``rank`` is
    the 1-based position of the new score if it made the list, else ``None``.
    """
    score = int(score)
    scores = load(path, top=10_000)      # load all, then re-trim
    scores.append(score)
    scores.sort(reverse=True)
    scores = scores[:top]

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scores, f)
    except OSError:
        logger.warning("Could not write highscore file: %s", path)

    rank = scores.index(score) + 1 if score in scores else None
    return scores, rank
