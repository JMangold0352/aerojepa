#!/usr/bin/env python3
"""Export a slim CPU release checkpoint: {config, model} only.

Usage:
  python scripts/export_release_ckpt.py
  python scripts/export_release_ckpt.py world_model
  python scripts/export_release_ckpt.py --out-dir /tmp/aerojepa-export

Writes released_weights/_export/<key>.pt (gitignored). Skips missing local
sources with a clear error and continues.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aerojepa.eval.weights import RELEASED_WEIGHTS, list_released_weights  # noqa: E402

# Registry key -> local training checkpoint (full trainer dump).
LOCAL_SRC = {
    "world_model": ROOT / "checkpoints" / "world_model" / "latest.pt",
    "real_finetune_fast": ROOT / "checkpoints" / "real_finetune_fast" / "latest.pt",
}

DEFAULT_OUT = ROOT / "released_weights" / "_export"


def export_one(key: str, out_dir: Path) -> Path | None:
    src = LOCAL_SRC[key]
    if not src.is_file():
        print(f"ERROR  {key}: missing local source {src} — skip", file=sys.stderr)
        return None

    import torch

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "config" not in ckpt or "model" not in ckpt:
        print(f"ERROR  {key}: unexpected checkpoint format in {src}", file=sys.stderr)
        return None

    slim = {"config": ckpt["config"], "model": ckpt["model"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{key}.pt"
    torch.save(slim, dest)
    src_mb = src.stat().st_size / (1024 * 1024)
    dst_mb = dest.stat().st_size / (1024 * 1024)
    print(f"OK  {key}: {src} ({src_mb:.1f} MiB) -> {dest} ({dst_mb:.1f} MiB)")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "keys",
        nargs="*",
        help="Registry keys (default: all). world_model / real_finetune_fast",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    args = parser.parse_args(argv)

    keys = args.keys or list_released_weights()
    unknown = [k for k in keys if k not in RELEASED_WEIGHTS]
    if unknown:
        print(f"ERROR  unknown keys: {unknown}. Known: {list(RELEASED_WEIGHTS)}", file=sys.stderr)
        return 1

    wrote = 0
    for key in keys:
        if export_one(key, args.out_dir) is not None:
            wrote += 1
    if wrote == 0:
        print("ERROR  no checkpoints exported", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
