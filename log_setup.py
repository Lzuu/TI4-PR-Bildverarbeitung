"""Logging fuer das Projekt: jede Programmausfuehrung schreibt eine eigene Logdatei.

Die Logdateien landen in logs/<name>_<datum>_<uhrzeit>.log (DEBUG und hoeher),
auf der Konsole erscheinen nur INFO-Meldungen und hoeher.
"""
import logging
import os
import time

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
KEEP_LOGS = 20           # aeltere Logdateien desselben Programms werden geloescht


def setup_logging(name, log_dir=LOG_DIR, keep=KEEP_LOGS):
    """Richtet das Logging ein und liefert den Pfad der neuen Logdatei."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)-7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    root.addHandler(file_handler)
    root.addHandler(console)

    remove_old_logs(log_dir, name, keep)
    return path


def remove_old_logs(log_dir, name, keep=KEEP_LOGS):
    """Loescht alle bis auf die neuesten `keep` Logdateien von `name`."""
    logs = sorted(f for f in os.listdir(log_dir) if f.startswith(name + "_") and f.endswith(".log"))
    for old in logs[:-keep] if keep > 0 else logs:
        try:
            os.remove(os.path.join(log_dir, old))
        except OSError:
            pass   # z.B. noch von einem anderen Prozess geoeffnet
