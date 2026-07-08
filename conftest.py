"""Make ``src/`` and the project root importable during tests.

``src`` so the ``aerojepa`` package resolves, and the project root so the
top-level ``visualizations`` and ``demo`` helper packages import too.
The research/prober side-project src is also added so its tests resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_PROBER_SRC = _ROOT / "research" / "prober" / "src"
for _path in (_ROOT / "src", _ROOT, _PROBER_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Headless-safe plotting so figure tests run without a display.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
