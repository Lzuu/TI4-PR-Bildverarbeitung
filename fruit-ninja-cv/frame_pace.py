"""How fast frames really arrive, so age budgets can follow a low frame rate (#88).

The age limits for model results (``POSE_MAX_AGE``, ``ASYNC_MAX_MASK_AGE``) are meant for
a game at around 30 frames per second. When the game runs at 3 frames per second -- as it
does on a slow machine with the suit on -- the result of the previous frame is already
0.3 s old when it arrives, so a fixed limit of 0.2 s or 0.3 s rejects every result and the
suit loses its pose, or the player's mask, in most frames: it flickers.

``FramePace`` keeps the *median* of the last few frame intervals. A single long frame (a
hiccup) does not move the median, a steadily slow frame rate does. The budget is then a
few frames long, never below the configured base and never above a cap, so a really old
result is still rejected.
"""

from collections import deque
from statistics import median


class FramePace:
    def __init__(self, window=8, warmup=4):
        self._intervals = deque(maxlen=window)
        self._last = None
        self.warmup = warmup

    def tick(self, now):
        """Call once per frame with the current time (seconds)."""
        if self._last is not None and now > self._last:
            self._intervals.append(now - self._last)
        if self._last is None or now > self._last:
            self._last = now

    @property
    def interval(self):
        """Typical time between two frames, or None until enough frames were seen."""
        if len(self._intervals) < self.warmup:
            return None
        return median(self._intervals)

    def budget(self, base, frames, cap):
        """``frames`` typical frames long, at least ``base`` and at most ``cap`` seconds
        (a ``cap`` below ``base`` never undercuts it)."""
        interval = self.interval
        if interval is None:
            return base
        return min(max(base, frames * interval), max(base, cap))
