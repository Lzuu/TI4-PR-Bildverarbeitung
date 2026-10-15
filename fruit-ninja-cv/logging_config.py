"""Central logging setup.

Writes a rotating logfile to ``logs/fruit-ninja.log`` so that whenever a problem
occurs there is a persistent trace to debug from. The console only shows
warnings and errors to keep gameplay output clean.
"""

import logging
import os
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = "fruit-ninja.log"


def setup_logging(level=logging.INFO):
    """Configure root logging (file + console) and return the logfile path."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, LOG_FILE)

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    # Guard against duplicate handlers if setup_logging() is called more than once
    if not root.handlers:
        file_handler = RotatingFileHandler(
            log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        file_handler.setFormatter(fmt)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

        console = logging.StreamHandler()
        console.setFormatter(fmt)
        console.setLevel(logging.WARNING)   # keep the terminal quiet during play
        root.addHandler(console)

    return log_path
