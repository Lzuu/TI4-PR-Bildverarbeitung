"""Optional bounded timing windows; log summaries, never individual frames."""
from collections import defaultdict, deque
from contextlib import contextmanager
import logging
import time

import numpy as np

logger=logging.getLogger(__name__)


class FrameMetrics:
    def __init__(self, enabled=False, interval=5.):
        self.enabled=enabled
        self.interval=interval
        self.samples=defaultdict(lambda:deque(maxlen=300))
        self._last=time.monotonic()
        self._frames=0

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        start=time.perf_counter()
        try:
            yield
        finally:
            self.samples[name].append((time.perf_counter()-start)*1000)

    def finish(self, started):
        if not self.enabled:
            return
        self.samples['frame_total'].append((time.perf_counter()-started)*1000)
        self._frames+=1
        now=time.monotonic()
        if now-self._last >= self.interval:
            summary={name:dict(median_ms=round(float(np.median(values)),2),
                               p95_ms=round(float(np.percentile(values,95)),2))
                     for name,values in self.samples.items() if values}
            logger.info('Live timing: %.1f frames/s; %s', self._frames/(now-self._last), summary)
            self._frames=0
            self._last=now
            self.samples.clear()
