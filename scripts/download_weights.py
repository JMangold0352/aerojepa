#!/usr/bin/env python3
"""CLI wrapper for released checkpoint downloads."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerojepa.eval.weights import main

if __name__ == "__main__":
    raise SystemExit(main())
