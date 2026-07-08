"""Make ``src/aerojepa`` importable when running scripts directly.

Importing this module (``import _bootstrap``) prepends the project ``src``
directory to ``sys.path``, so the scripts work from a plain checkout without
first running ``pip install -e .``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
# ``src`` for the installable package; the project root so the top-level
# ``visualizations`` and ``demo`` helper packages import too.
for _path in (_SRC, _ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
