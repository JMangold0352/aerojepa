#!/usr/bin/env python
"""Convert The Wilds Drones (Parrot ANAFI) data into AeroJEPA training format.

Reads ``data/raw/thewilds/`` (HuggingFace: imageomics/thewilds_drones) and writes
symlinked MP4s + ACTION_COLUMNS CSVs to ``data/flights/``.

Examples::

    # After downloading the dataset:
    python scripts/convert_wilds.py --raw-dir data/raw/thewilds --out-dir data/flights

    # Copy videos instead of symlinking:
    python scripts/convert_wilds.py --copy
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from aerojepa.data.wilds import convert_wilds, discover_wilds_videos


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/thewilds")
    parser.add_argument("--out-dir", default="data/flights")
    parser.add_argument("--copy", action="store_true", help="Copy MP4s instead of symlinking.")
    args = parser.parse_args()

    videos = discover_wilds_videos(args.raw_dir)
    if not videos:
        raise SystemExit(
            f"No videos found under {args.raw_dir}. "
            "Download first: huggingface-cli download imageomics/thewilds_drones "
            "--repo-type dataset --local-dir data/raw/thewilds"
        )

    print(f"Found {len(videos)} Wilds clip(s) under {args.raw_dir}")
    written = convert_wilds(args.raw_dir, args.out_dir, link_videos=not args.copy)
    print(f"Wrote {len(written)} clip(s) to {args.out_dir}")
    print("Next:")
    print("  python scripts/preprocess_real.py --probe --input-dir", args.out_dir)
    print("  ./scripts/launch_training.sh configs/aerojepa_finetune.yaml")


if __name__ == "__main__":
    main()
