#!/usr/bin/env python
"""AeroJEPA Gradio demo entry point.

    python app.py                                  # try world_model download / local, else smoke
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


def _resolve_default_checkpoint() -> str | None:
    """Prefer local world_model; try download; else untrained smoke."""
    local = _ROOT / "checkpoints" / "world_model" / "latest.pt"
    if local.is_file():
        return str(local)
    try:
        from aerojepa.eval.weights import ensure_checkpoint

        return str(ensure_checkpoint("world_model", pretrained=True))
    except Exception:
        print("weights are not published yet; running untrained smoke model")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint (optional).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    args = parser.parse_args()
    checkpoint = args.checkpoint
    if not checkpoint:
        checkpoint = _resolve_default_checkpoint()
    launch_demo(checkpoint=checkpoint, share=args.share)


if __name__ == "__main__":
    main()
