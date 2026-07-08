#!/usr/bin/env python
"""Head-to-head comparison of two checkpoints on latent prediction.

Typical use: baseline (feed-forward) vs looped (recurrent) predictor, to
quantify what recurrence buys on the same held-out clips.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from aerojepa.eval import load_model
from aerojepa.eval.latent_pred import latent_prediction_metrics
from aerojepa.masking import build_mask_collator
from aerojepa.train import build_dataloaders_from_cfg
from aerojepa.utils.device import get_device


def _score(checkpoint: str, device, max_batches: int) -> dict:
    model, cfg = load_model(checkpoint, device)
    _, val_loader = build_dataloaders_from_cfg(cfg)
    grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = build_mask_collator(cfg, grid, model.encoder.num_temporal)
    return latent_prediction_metrics(model, val_loader, collator, device, cfg, max_batches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--looped-checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--out", default="results/comparison.json")
    args = parser.parse_args()

    device = get_device(args.device)
    baseline = _score(args.baseline_checkpoint, device, args.max_batches)
    looped = _score(args.looped_checkpoint, device, args.max_batches)
    delta = looped["cosine"] - baseline["cosine"]

    print(f"baseline cosine : {baseline['cosine']:.4f}")
    print(f"looped   cosine : {looped['cosine']:.4f}")
    print(f"delta           : {delta:+.4f}")

    report = {"baseline": baseline, "looped": looped, "cosine_delta": delta}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
