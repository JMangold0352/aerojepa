#!/usr/bin/env python
"""Standardize real footage into AeroJEPA's training format.

Converts any videos (Tello captures, phone clips, downloaded drone footage) to a
consistent codec + frame rate, optionally square-cropped/resized, keeping any
telemetry aligned. Also a "dataset doctor" that probes a folder and reports
per-clip frame counts, fps, resolution, and telemetry status.

Examples::

    # Inspect what you have (no writes):
    python scripts/preprocess_real.py --probe --input-dir data/raw

    # Standardize raw footage -> data/flights/ at 15 fps, center-square:
    python scripts/preprocess_real.py --input-dir data/raw \
        --output-dir data/flights --target-fps 15 --square

    # Trim long clips and downscale to 128x128 to shrink files:
    python scripts/preprocess_real.py --input-dir data/raw \
        --max-seconds 20 --square --resize 128
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401


def _print_probe_table(infos) -> int:
    print(f"{'clip':40s} {'frames':>7} {'fps':>6} {'dur(s)':>7} {'WxH':>11} {'tel':>8}  issues")
    print("-" * 100)
    total_issues = 0
    for info in infos:
        wh = f"{info.width}x{info.height}"
        issues = "; ".join(info.issues)
        total_issues += len(info.issues)
        print(
            f"{info.path.name:40s} {info.frames:7d} {info.fps:6.1f} {info.duration_s:7.1f} "
            f"{wh:>11} {info.telemetry:>8}  {issues}"
        )
    print("-" * 100)
    print(f"{len(infos)} clip(s), {total_issues} issue(s).")
    return total_issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input-dir", required=True, help="Folder of source videos.")
    parser.add_argument("--output-dir", default="data/flights")
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--max-seconds", type=float, default=None, help="Trim clips to N seconds.")
    parser.add_argument("--square", action="store_true", help="Center-crop to a square frame.")
    parser.add_argument("--resize", type=int, default=None, help="Resize to NxN pixels.")
    parser.add_argument("--prefix", default="clip", help="Output filename prefix.")
    parser.add_argument("--probe", action="store_true", help="Only inspect; do not write.")
    args = parser.parse_args()

    from aerojepa.data.preprocess import (
        PreprocessConfig,
        preprocess_directory,
        probe_directory,
    )

    if args.probe:
        infos = probe_directory(args.input_dir)
        if not infos:
            print(f"No videos found in {args.input_dir}")
            return
        _print_probe_table(infos)
        return

    print(f"Standardizing {args.input_dir} -> {args.output_dir} @ {args.target_fps} fps")
    results = preprocess_directory(
        PreprocessConfig(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_fps=args.target_fps,
            max_seconds=args.max_seconds,
            square=args.square,
            resize=args.resize,
            prefix=args.prefix,
        )
    )
    print(f"\nWrote {len(results)} clip(s) to {args.output_dir}")
    print("Next: python scripts/train.py --config configs/aerojepa_finetune.yaml")


if __name__ == "__main__":
    main()
