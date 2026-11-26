"""Exclusive inference owner with latest-only mailboxes and source fencing."""
from collections import OrderedDict
from dataclasses import dataclass
import logging
import threading
import time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Result:
    value: object
    frame: object
    captured_at: float
    seconds: float
    sequence: int


class LatestWorker:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __init__(self):
        self._condition = threading.Condition()
        self._pending = OrderedDict()
        self._keys, self._results, self._failed = {}, {}, set()
        self._epochs = {}
        self._busy = self._closed = False
        self._sequence = 0
        self._thread = threading.Thread(target=self._run, name='cv-inference', daemon=True)
        self._thread.start()

    def activate(self, kind, key):
        with self._condition:
            if self._keys.get(kind) != key:
                self._keys[kind] = key
                self._epochs[kind] = self._epochs.get(kind, 0) + 1
                self._pending.pop(kind, None)
                self._results.pop(kind, None)

    def submit(self, kind, key, frame, infer, captured_at, due=True):
        self.activate(kind, key)
        with self._condition:
            if self._closed or kind in self._failed:
                return
            if not due and kind not in self._pending:
                return
            self._sequence += 1
            self._pending[kind] = (key, self._epochs[kind], frame.copy(), infer, captured_at, self._sequence)
            self._condition.notify()

    def poll(self, kind, key):
        with self._condition:
            return self._results.get(kind) if self._keys.get(kind) == key else None

    def failed(self, kind):
        with self._condition:
            return kind in self._failed

    def retry(self, kind):
        with self._condition:
            self._epochs[kind] = self._epochs.get(kind, 0) + 1
            self._failed.discard(kind)
            self._pending.pop(kind, None)
            self._results.pop(kind, None)

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._pending)
                if self._closed:
                    return
                kind, (key, epoch, frame, infer, captured, sequence) = self._pending.popitem(last=False)
                self._busy = True
            started = time.perf_counter()
            try:
                value = infer(frame)
                result = Result(value, frame, captured, time.perf_counter()-started, sequence)
            except Exception:
                with self._condition:
                    if not self._closed and self._epochs[kind] == epoch:
                        logger.error('Inference disabled: %s', kind, exc_info=True)
                        self._failed.add(kind)
                        self._pending.pop(kind, None)
                        self._results.pop(kind, None)
            else:
                with self._condition:
                    if not self._closed and self._keys.get(kind) == key and self._epochs[kind] == epoch:
                        self._results[kind] = result
            finally:
                with self._condition:
                    self._busy = False
                    self._condition.notify_all()

    def wait_idle(self, timeout):
        """Bounded test synchronization; never used by the game loop."""
        with self._condition:
            return self._condition.wait_for(lambda: not self._busy and not self._pending, timeout)

    def close(self):
        # Native forward cannot safely be cancelled; do not join it on the UI.
        with self._condition:
            self._closed = True
            self._pending.clear()
            self._results.clear()
            self._condition.notify_all()
