"""Summarise the cut decisions of the challenge mode from the game log (#33).

The challenge writes one log line per judged cut (not per frame)::

    Cut ok: direction=right speed=2.10 deviation=8
    Cut wrong_direction: direction=left speed=2.80 deviation=172
    Cut too_slow: direction=down speed=0.45 deviation=12

``speed`` is in field widths per second, ``deviation`` is the angle to the arrow in
degrees. Play the prepared cut tasks (see README, "Praktischer Versuch"), then run::

    python tools/analyze_cuts.py logs/fruit-ninja.log

to get counts and ranges per outcome next to the current thresholds
(``CUT_MIN_SPEED``, ``CUT_TOLERANCE_DEG``). The margin between the slowest accepted cut
and the fastest rejected slow touch (and between the largest accepted and the smallest
rejected angle) is what the thresholds should be justified with.
"""

import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

LINE = re.compile(r"Cut (ok|wrong_direction|too_slow): direction=(\w+) "
                  r"speed=([\d.]+) deviation=([\d.]+)")


def parse_log(lines):
    """One record per logged cut decision; every other log line is ignored."""
    records = []
    for line in lines:
        m = LINE.search(line)
        if m:
            records.append({"result": m.group(1), "direction": m.group(2),
                            "speed": float(m.group(3)), "deviation": float(m.group(4))})
    return records


def summarize(records):
    """``{result: {count, speed_min/median/max, deviation_min/median/max}}``."""
    summary = {}
    for result in sorted({r["result"] for r in records}):
        rows = [r for r in records if r["result"] == result]
        speeds = [r["speed"] for r in rows]
        devs = [r["deviation"] for r in rows]
        summary[result] = {
            "count": len(rows),
            "speed_min": min(speeds), "speed_median": statistics.median(speeds),
            "speed_max": max(speeds),
            "deviation_min": min(devs), "deviation_median": statistics.median(devs),
            "deviation_max": max(devs),
        }
    return summary


def report(summary):
    if not summary:
        return "Keine Schnitt-Eintraege im Log gefunden (erst eine Challenge spielen)."
    lines = [f"Aktuelle Grenzwerte: CUT_MIN_SPEED = {config.CUT_MIN_SPEED} Spielfeldbreiten/s, "
             f"CUT_TOLERANCE_DEG = {config.CUT_TOLERANCE_DEG} Grad", "",
             f"{'Ergebnis':16s} {'Anzahl':>6s}  {'Tempo min/median/max':>24s}  "
             f"{'Abweichung min/median/max':>27s}"]
    for result, d in summary.items():
        lines.append(f"{result:16s} {d['count']:6d}  "
                     f"{d['speed_min']:7.2f} /{d['speed_median']:6.2f} /{d['speed_max']:6.2f}  "
                     f"{d['deviation_min']:8.0f} /{d['deviation_median']:6.0f} /{d['deviation_max']:6.0f}")
    lines.append("")
    ok, slow, wrong = summary.get("ok"), summary.get("too_slow"), summary.get("wrong_direction")
    if ok and slow:
        lines.append(f"Tempo-Abstand: langsamster akzeptierter Schnitt {ok['speed_min']:.2f}, "
                     f"schnellste abgelehnte langsame Beruehrung {slow['speed_max']:.2f} "
                     f"(Grenzwert dazwischen: {config.CUT_MIN_SPEED}).")
    if ok and wrong:
        lines.append(f"Winkel-Abstand: groesste akzeptierte Abweichung {ok['deviation_max']:.0f} "
                     f"Grad, kleinste abgelehnte {wrong['deviation_min']:.0f} Grad "
                     f"(Grenzwert dazwischen: {config.CUT_TOLERANCE_DEG}).")
    return "\n".join(lines)


def main(argv):
    path = Path(argv[1] if len(argv) > 1 else "logs/fruit-ninja.log")
    with open(path, encoding="utf-8", errors="replace") as f:
        print(report(summarize(parse_log(f))))


if __name__ == "__main__":
    main(sys.argv)
