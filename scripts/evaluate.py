#!/usr/bin/env python
"""Evaluate a trained AeroJEPA checkpoint and write a metrics report.

Reports latent-prediction quality, multi-step rollout error, and (for looped
models) per-loop refinement and exit-gate behavior. Results are printed and
saved as JSON under ``results/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from aerojepa.eval import load_model
from aerojepa.eval.latent_pred import latent_prediction_metrics
from aerojepa.eval.loop_metrics import loop_metrics
from aerojepa.eval.rollout import rollout_metrics
from aerojepa.masking import build_mask_collator
from aerojepa.models.looped_predictor import LoopedVideoPredictor
from aerojepa.train import build_dataloaders_from_cfg
from aerojepa.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=None, help="Output JSON path (default: results/<run>_eval.json)")
    parser.add_argument("--max-batches", type=int, default=8)
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    _, val_loader = build_dataloaders_from_cfg(cfg)
    grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = build_mask_collator(cfg, grid, model.encoder.num_temporal)

    report: dict = {"checkpoint": args.checkpoint, "objective": cfg.get("objective", "masked")}

    latent = latent_prediction_metrics(model, val_loader, collator, device, cfg, args.max_batches)
    report["latent_prediction"] = latent
    print(f"latent cosine={latent['cosine']:.4f}  smooth_l1={latent['smooth_l1']:.4f}")

    roll = rollout_metrics(model, val_loader, device, cfg, max_batches=args.max_batches)
    report["rollout"] = roll
    print("rollout cosine by horizon:", [f"{c:.3f}" for c in roll["cosine"]])

    if isinstance(model.predictor, LoopedVideoPredictor):
        loops = loop_metrics(model, val_loader, collator, device, cfg, args.max_batches)
        report["loop_analysis"] = loops
        print(f"expected loops={loops['expected_loops']:.2f}  per-loop cosine={[f'{c:.3f}' for c in loops['per_loop_cosine']]}")

    out_path = Path(args.out) if args.out else Path("results") / (Path(cfg["train"]["run_dir"]).name + "_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
