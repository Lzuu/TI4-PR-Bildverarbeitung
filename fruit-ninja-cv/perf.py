"""Frame-time diagnostics: real frame rate and per-stage times (#42, PERF-06).

The game loop marks each frame and times its stages (camera read, tracking,
segmentation, suit, drawing, display ...). Every ``interval`` seconds the
window is summarised -- frames/s, median and P95 of the frame time and of
every stage, counters (e.g. failed reads) and maxima (e.g. the age of a
reused YOLO mask) -- logged as one line and optionally appended to a CSV.
Nothing is logged per frame.

A stage only counts in the frames where it actually ran, so an expensive step
that runs every few frames (YOLO inference) is not hidden by cheap frames.

What this measures is *computation and waiting inside the loop*. The time
spent in ``read`` is how long the loop waited for the camera; a read that
returns at once means a buffered (older) frame. The real camera-to-screen
latency also contains exposure, USB transfer and display time and can only be
measured from outside (see Architektur.md).
"""

import csv
import logging
import os
import time
from collections import defaultdict
from contextlib import contextmanager

logger = logging.getLogger(__name__)

CSV_FIELDS = ["time", "frames", "fps", "stage", "count", "median_ms", "p95_ms", "max_ms"]


def percentile(values, q):
    """``q``-th percentile (linear interpolation) of ``values``, or None if empty."""
    if not values:
        return None
    s = sorted(values)
    pos = (len(s) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def _describe(values_ms):
    return {"count": len(values_ms), "median": percentile(values_ms, 50),
            "p95": percentile(values_ms, 95), "max": max(values_ms)}


class FrameStats:
    def __init__(self, interval=10.0, clock=time.perf_counter, log=logger.info, csv_path=None):
        self.interval = interval
        self._clock = clock
        self._log = log
        self.csv_path = csv_path
        self.last_summary = None
        self._reset_window(clock())
        self._frame_start = None

    def _reset_window(self, now):
        self._window_start = now
        self._frames = []
        self._stages = defaultdict(list)
        self._counters = defaultdict(int)
        self._maxima = {}

    # -- recording ----------------------------------------------------------
    def start_frame(self):
        self._frame_start = self._clock()

    @contextmanager
    def stage(self, name):
        t0 = self._clock()
        try:
            yield
        finally:
            self.add(name, self._clock() - t0)

    def add(self, name, seconds):
        self._stages[name].append(seconds * 1000.0)

    def count(self, name, n=1):
        self._counters[name] += n

    def maximum(self, name, value):
        if name not in self._maxima or value > self._maxima[name]:
            self._maxima[name] = value

    def end_frame(self):
        """Close the frame; returns the window summary when ``interval`` is over."""
        now = self._clock()
        if self._frame_start is not None:
            self._frames.append((now - self._frame_start) * 1000.0)
            self._frame_start = None
        elapsed = now - self._window_start
        if elapsed < self.interval or not self._frames:
            return None
        summary = {
            "frames": len(self._frames),
            "seconds": elapsed,
            "fps": len(self._frames) / elapsed,
            "frame_ms": _describe(self._frames),
            "stages": {k: _describe(v) for k, v in self._stages.items() if v},
            "counters": dict(self._counters),
            "maxima": dict(self._maxima),
        }
        self.last_summary = summary
        self._report(summary)
        self._reset_window(now)
        return summary

    # -- output ---------------------------------------------------------------
    def _report(self, s):
        if self._log is not None:
            stages = ", ".join(f"{k} {v['median']:.1f}/{v['p95']:.1f}"
                               + (f" (n={v['count']})" if v["count"] != s["frames"] else "")
                               for k, v in sorted(s["stages"].items()))
            extra = "".join(f", {k}={v}" for k, v in sorted(s["counters"].items()))
            extra += "".join(f", max {k}={v:.0f}" for k, v in sorted(s["maxima"].items()))
            self._log("Perf %.1f fps (%d frames), frame %.1f/%.1f ms median/P95 | %s%s",
                      s["fps"], s["frames"], s["frame_ms"]["median"], s["frame_ms"]["p95"],
                      stages, extra)
        if self.csv_path:
            self._write_csv(s)

    def _write_csv(self, s):
        try:
            folder = os.path.dirname(self.csv_path)
            if folder:
                os.makedirs(folder, exist_ok=True)
            new = not os.path.exists(self.csv_path)
            with open(self.csv_path, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
                if new:
                    writer.writeheader()
                stamp = time.strftime("%Y-%m-%d %H:%M:%S")
                for name, d in [("frame", s["frame_ms"])] + sorted(s["stages"].items()):
                    writer.writerow({"time": stamp, "frames": s["frames"],
                                     "fps": f"{s['fps']:.2f}", "stage": name,
                                     "count": d["count"], "median_ms": f"{d['median']:.2f}",
                                     "p95_ms": f"{d['p95']:.2f}", "max_ms": f"{d['max']:.2f}"})
        except OSError:
            logger.warning("Could not write perf CSV %s -- export off", self.csv_path,
                           exc_info=True)
            self.csv_path = None

    def overlay_lines(self):
        """Short text lines of the last summary for an on-screen overlay."""
        s = self.last_summary
        if s is None:
            return []
        lines = [f"{s['fps']:.1f} FPS  Frame {s['frame_ms']['median']:.0f}/"
                 f"{s['frame_ms']['p95']:.0f} ms (Median/P95)"]
        for k, v in sorted(s["stages"].items(), key=lambda kv: -kv[1]["median"]):
            lines.append(f"{k}: {v['median']:.1f}/{v['p95']:.1f} ms"
                         + (f"  n={v['count']}" if v["count"] != s["frames"] else ""))
        for k, v in sorted(s["counters"].items()):
            lines.append(f"{k}: {v}")
        for k, v in sorted(s["maxima"].items()):
            lines.append(f"max {k}: {v:.0f}")
        return lines
