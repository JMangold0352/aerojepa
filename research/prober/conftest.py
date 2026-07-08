"""Test setup for the AeroProber package.

Puts ``research/prober/src`` on sys.path so ``aerojepa_research.prober``
resolves, and the project root + its ``src`` so the parent ``aerojepa``
package (frozen world model) is importable for integration tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROBER_ROOT = _HERE.parent
_PROJECT_ROOT = _PROBER_ROOT.parent.parent

for _path in (_PROBER_ROOT / "src", _PROJECT_ROOT / "src", _PROJECT_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
