#!/usr/bin/env python
"""AeroJEPA Gradio demo entry point.

    python app.py                                  # untrained smoke model
    python app.py --checkpoint checkpoints/world_model/latest.pt

This file is also the Hugging Face Spaces entry point.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``src/`` and the top-level ``demo`` package importable from a checkout.
_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from demo.render import launch_demo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint (optional).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    args = parser.parse_args()
    launch_demo(checkpoint=args.checkpoint, share=args.share)


if __name__ == "__main__":
    main()
