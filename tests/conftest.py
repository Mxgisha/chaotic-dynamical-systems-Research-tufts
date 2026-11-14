"""Make the repository root importable when pytest is run from anywhere."""

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
