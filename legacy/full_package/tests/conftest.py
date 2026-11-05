"""Make the in-repo package importable and keep matplotlib headless."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The sandboxed home is read-only; give matplotlib a writable config dir so the
# figure tests can run at all.
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".agent_scratch" / "mplcache"))
os.environ.setdefault("LORENZ63_BACKEND", "numpy")

# Force a headless backend *before* pyplot is imported anywhere, otherwise the
# figure tests try to reach for a display and fail with a _tkinter error.
import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)
