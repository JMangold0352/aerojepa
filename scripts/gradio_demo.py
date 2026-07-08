#!/usr/bin/env python
"""Launch the AeroJEPA Gradio demo (equivalent to ``python app.py``)."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from demo.render import launch_demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Optional trained checkpoint to load.")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    launch_demo(checkpoint=args.checkpoint, share=args.share)


if __name__ == "__main__":
    main()
