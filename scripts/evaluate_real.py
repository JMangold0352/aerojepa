#!/usr/bin/env python
"""Evaluate a checkpoint on real footage and report the synthetic-vs-real gap.

Runs the *same* metrics (latent cosine, smooth-L1, multi-step rollout) on two
data sources with identical code:

  1. the synthetic benchmark the model was trained/embedded with, and
  2. a folder of real clips (``--data-dir``, default ``data/flights``),

then prints and saves the difference. A small gap means the world model
transfers; a large one shows how much real data (or fine-tuning) is still needed.

Examples::

    python scripts/evaluate_real.py --checkpoint checkpoints/world_model/latest.pt
    python scripts/evaluate_real.py \
        --checkpoint checkpoints/real_finetune/latest.pt \
        --data-dir data/flights --out results/world_model_real_eval.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from aerojepa.eval import load_model
from aerojepa.eval.real_gap import evaluate_real_gap
from aerojepa.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/flights", help="Folder of real clips.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="Output JSON path.")
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument(
        "--window-mode", default="uniform", choices=["uniform", "sliding"],
        help="How to sample the real clips for evaluation.",
    )
    args = parser.parse_args()

    print("=== synthetic ===")
    report = evaluate_real_gap(
        args.checkpoint,
        args.data_dir,
        device=args.device,
        max_batches=args.max_batches,
        window_mode=args.window_mode,
    )
    synth = report["synthetic"]
    real = report["real"]
    cos_gap = report["gap"]["latent_cosine"]
    print(f"latent cosine={synth['latent_prediction']['cosine']:.4f}")
    print(f"=== real ({args.data_dir}) ===")
    print(f"latent cosine={real['latent_prediction']['cosine']:.4f}")
    print(f"\nsim-to-real latent-cosine gap: {cos_gap:+.4f} "
          f"(synthetic {synth['latent_prediction']['cosine']:.4f} -> "
          f"real {real['latent_prediction']['cosine']:.4f})")

    _, cfg = load_model(args.checkpoint, get_device(args.device))
    run_name = Path(cfg["train"]["run_dir"]).name
    out_path = Path(args.out) if args.out else Path("results") / f"{run_name}_real_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
