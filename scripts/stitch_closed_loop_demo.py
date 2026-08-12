#!/usr/bin/env python
"""Stitch closed-loop hover / waypoint / recover GIFs into one demo reel.

Looks under ``visualizations/closed_loop/`` for planner GIFs (and a random
failure clip when present), writes:

  * ``docs/gallery/closed_loop_demo_reel.gif``
  * ``visualizations/closed_loop/closed_loop_demo_reel.gif``

Example::

    python scripts/stitch_closed_loop_demo.py
    python scripts/stitch_closed_loop_demo.py --include-random
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from aerojepa.sim.closed_loop import stitch_demo_reel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", default="visualizations/closed_loop")
    parser.add_argument("--out", default="docs/gallery/closed_loop_demo_reel.gif")
    parser.add_argument("--also-copy", default="visualizations/closed_loop/closed_loop_demo_reel.gif")
    parser.add_argument("--include-random", action="store_true",
                        help="Append a random-policy waypoint fail clip for contrast.")
    parser.add_argument("--panel-size", type=int, nargs=2, default=(256, 256), metavar=("W", "H"))
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    specs: list[tuple[str, Path]] = []
    mapping = [
        ("1. Hover", in_dir / "closed_loop_hover_planner.gif"),
        ("2. Waypoint", in_dir / "closed_loop_waypoint_planner.gif"),
        ("3. Recover", in_dir / "closed_loop_recover_planner.gif"),
    ]
    for title, path in mapping:
        if path.exists():
            specs.append((title, path))
        else:
            print(f"skip missing: {path}")

    if args.include_random:
        rnd = in_dir / "closed_loop_waypoint_random.gif"
        if rnd.exists():
            specs.append(("Contrast: random", rnd))

    if not specs:
        raise SystemExit(
            f"No GIFs found in {in_dir}. Run run_closed_loop_demo.py for hover/waypoint/recover first."
        )

    out = stitch_demo_reel(specs, args.out, panel_size=tuple(args.panel_size))  # type: ignore[arg-type]
    print(f"reel -> {out}")
    if args.also_copy:
        also = stitch_demo_reel(specs, args.also_copy, panel_size=tuple(args.panel_size))  # type: ignore[arg-type]
        print(f"copy -> {also}")


if __name__ == "__main__":
    main()
