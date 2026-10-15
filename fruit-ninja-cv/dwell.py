"""VR-style dwell selection.

Keep the marker on a target for ``hold_seconds`` and the selection confirms.
A circular progress ring (drawn in main) fills up while the marker stays put.
Timing is wall-clock based, so it is independent of the frame rate.
"""

import time


class DwellTracker:
    def __init__(self, hold_seconds=3.0, time_fn=time.time):
        self.hold = hold_seconds
        self._time = time_fn
        self.active = None       # index of the target currently being dwelled on
        self._start = 0.0

    def reset(self):
        self.active = None
        self._start = 0.0

    @staticmethod
    def hit(point, targets):
        """Index of the first target circle containing ``point``, else None."""
        if point is None:
            return None
        px, py = point
        for i, (cx, cy, r) in enumerate(targets):
            if (px - cx) ** 2 + (py - cy) ** 2 <= r * r:
                return i
        return None

    def update(self, point, targets):
        """Advance dwell state.

        Returns ``(active_index, progress, confirmed_index)`` where progress is
        in 0..1 and confirmed_index is set once the target is held long enough.
        """
        idx = self.hit(point, targets)
        now = self._time()
        if idx is None or idx != self.active:
            self.active = idx
            self._start = now
        if self.active is None:
            return None, 0.0, None
        progress = min((now - self._start) / self.hold, 1.0)
        confirmed = self.active if progress >= 1.0 else None
        return self.active, progress, confirmed
