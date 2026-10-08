"""Make the game modules importable from the tests directory."""

import os
import sys

# Add the project root (one level up) to the import path so tests can do
# `import config`, `import game`, etc. without installing a package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
