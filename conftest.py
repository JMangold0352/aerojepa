"""Make ``src/`` and the project root importable during tests.

``src`` so the ``aerojepa`` package resolves, and the project root so the
top-level ``visualizations`` and ``demo`` helper packages import too.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (_ROOT / "src", _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

# Headless-safe plotting so figure tests run without a display.
import matplotlib  # noqa: E402

matplotlib.use("Agg")
