"""VR-style dwell selection.

Keep the marker on a target for ``hold_seconds`` and the selection confirms.
A circular progress ring (drawn in main) fills up while the marker stays put.
Timing is wall-clock based, so it is independent of the frame rate.
"""

import time


class DwellTracker:
    def __init__(self, hold_seconds=3.0, time_fn=time.time, lost_unlock_seconds=0.5):
        self.hold = hold_seconds
        self.lost_unlock = lost_unlock_seconds   # a lost marker lifts the lock only after this long
        self._time = time_fn
        self.active = None       # index of the target currently being dwelled on
        self._start = 0.0
        self._locked = False     # waiting for the marker to leave every target
        self._lost_since = None  # while locked: when the marker was lost

    def reset(self, require_exit=False):
        """Forget the current dwell.

        With ``require_exit`` the marker first has to leave every target before a
        new dwell can start. Used when the screen changes: the targets of the next
        screen can lie where the pen still rests, and that must not confirm
        anything by itself (#63). A marker that is lost only counts as gone after
        ``lost_unlock_seconds`` -- a single frame in which the detection drops out
        must not lift the lock while the pen rests on a target.
        """
        self.active = None
        self._start = 0.0
        self._locked = require_exit
        self._lost_since = None

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
        if self._locked:
            if point is None:                      # lost: gone for good, or just a dropout?
                if self._lost_since is None:
                    self._lost_since = now
                if now - self._lost_since < self.lost_unlock:
                    return None, 0.0, None
            else:
                self._lost_since = None
                if idx is not None:                # still (or again) on a target
                    return None, 0.0, None
            self._locked = False
        if idx is None or idx != self.active:
            self.active = idx
            self._start = now
        if self.active is None:
            return None, 0.0, None
        progress = min((now - self._start) / self.hold, 1.0)
        confirmed = self.active if progress >= 1.0 else None
        return self.active, progress, confirmed
