#!/usr/bin/env python
"""Render the AeroJEPA figure suite from a trained checkpoint.

Thin wrapper around ``visualizations/generate_all_figures.py`` so figures can be
regenerated with a single command.
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from visualizations.generate_all_figures import generate_all  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="visualizations/figures")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fast", action="store_true", help="Fewer clips for a quick smoke render.")
    args = parser.parse_args()
    generate_all(args.checkpoint, args.out_dir, device=args.device, fast=args.fast)


if __name__ == "__main__":
    main()
