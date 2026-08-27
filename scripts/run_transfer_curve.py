#!/usr/bin/env python
"""Fine-tune world_model on increasing real-data volumes; plot sim-to-real transfer.

Holds out a fixed eval set (never used for training), fine-tunes from the
synthetic ``world_model`` checkpoint on 1 / 5 / N clips, and measures:

  - sim-to-real latent cosine gap on held-out real clips
  - rollout quality @ horizon 4 on the same holdout

Results: ``results/transfer_curve/summary.json`` (+ per-run JSON).
Figure: ``results/transfer_curve/transfer_curve.png`` (and PDF).

Examples::

    # Full curve (~15-25 min on MPS with 5 epochs per point):
    python scripts/run_transfer_curve.py --device mps

    # Quick smoke (1 clip, 1 epoch):
    python scripts/run_transfer_curve.py --sizes 1 --epochs 1 --device mps

    # Re-plot from existing summary:
    python scripts/run_transfer_curve.py --plot-only

    # Re-score checkpoints without retraining:
    python scripts/run_transfer_curve.py --eval-only
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from aerojepa.eval.real_gap import evaluate_real_gap
from aerojepa.eval.transfer_curve import (
    build_manifest,
    prepare_subset_dirs,
    save_manifest,
    summarize_eval_point,
)
from aerojepa.train import train
from aerojepa.utils.config import load_config
from aerojepa.utils.device import get_device


def _parse_sizes(text: str) -> list[int]:
    return sorted({int(x.strip()) for x in text.split(",") if x.strip()})


def _plot_summary(summary_path: Path, out_dir: Path) -> None:
    from visualizations.plot_transfer_curve import plot_transfer_curve

    plot_transfer_curve(summary_path, out_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/flights_128")
    parser.add_argument("--config", default="configs/aerojepa_transfer_curve.yaml")
    parser.add_argument(
        "--init-checkpoint",
        default="checkpoints/world_model/latest.pt",
        help="Synthetic pretrain to fine-tune from.",
    )
    parser.add_argument("--sizes", default="1,5,15", help="Comma-separated train clip counts.")
    parser.add_argument("--holdout", type=int, default=3, help="Clips held out for eval only.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--out-dir", default="results/transfer_curve")
    parser.add_argument("--data-root", default="data/transfer_curve")
    parser.add_argument("--clean-data", action="store_true", help="Rebuild symlink subset dirs.")
    parser.add_argument("--train-only", action="store_true", help="Skip baseline eval and plotting.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training; eval existing ckpts.")
    parser.add_argument("--plot-only", action="store_true", help="Render figure from summary.json.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"

    if args.plot_only:
        if not summary_path.is_file():
            raise SystemExit(f"Missing {summary_path}. Run the experiment first.")
        _plot_summary(summary_path, out_dir)
        print(f"Wrote {out_dir / 'transfer_curve.png'}")
        return

    device = get_device(args.device)
    sizes = _parse_sizes(args.sizes)
    manifest = build_manifest(
        args.source_dir, holdout_count=args.holdout, train_sizes=sizes,
    )
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest["init_checkpoint"] = args.init_checkpoint
    manifest["epochs_per_run"] = args.epochs
    save_manifest(manifest, out_dir / "manifest.json")

    data_dirs = prepare_subset_dirs(
        manifest, args.data_root, clean=args.clean_data,
    )
    eval_dir = data_dirs["eval_holdout"]
    print(f"[transfer] eval holdout ({len(manifest['eval_holdout'])} clips): {eval_dir}")

    eval_kw = dict(device=args.device, max_batches=args.max_batches)
    points: list[dict] = []

    if not args.train_only:
        print("\n=== baseline (synthetic world_model, no real fine-tune) ===")
        base_report = evaluate_real_gap(args.init_checkpoint, eval_dir, **eval_kw)
        base_point = summarize_eval_point(
            0, "synthetic only", args.init_checkpoint, base_report,
        )
        base_point["train_clip_names"] = []
        points.append(base_point)
        (out_dir / "baseline_eval.json").write_text(json.dumps(base_report, indent=2))
        print(
            f"  real cosine={base_point['real_latent_cosine']:.4f}  "
            f"gap={base_point['sim_to_real_gap']:+.4f}  "
            f"rollout@4={base_point['rollout_cosine_h4']:.4f}"
        )

    for n in manifest["train_sizes"]:
        ckpt_path = Path(f"checkpoints/transfer_curve/n{n}/latest.pt")
        train_dir = data_dirs[f"train_n{n}"]
        clip_names = manifest["subsets"][str(n)]

        if not args.eval_only:
            print(f"\n=== fine-tune on {n} clip(s): {clip_names} ===")
            cfg = load_config(args.config)
            cfg = copy.deepcopy(cfg)
            cfg["data"]["data_dir"] = str(train_dir)
            if n <= 2:
                cfg["data"]["val_fraction"] = 0.5
            cfg["train"]["epochs"] = args.epochs
            cfg["train"]["run_dir"] = f"runs/transfer_curve/n{n}"
            cfg["train"]["checkpoint_dir"] = f"checkpoints/transfer_curve/n{n}"
            train(cfg, device, init_checkpoint=args.init_checkpoint)

        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Expected checkpoint after training: {ckpt_path}")

        report = evaluate_real_gap(str(ckpt_path), eval_dir, **eval_kw)
        point = summarize_eval_point(n, f"{n} clips", str(ckpt_path), report)
        point["train_clip_names"] = clip_names
        points.append(point)
        (out_dir / f"n{n}_eval.json").write_text(json.dumps(report, indent=2))
        print(
            f"  real cosine={point['real_latent_cosine']:.4f}  "
            f"gap={point['sim_to_real_gap']:+.4f}  "
            f"rollout@4={point['rollout_cosine_h4']:.4f}"
        )

    summary = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "source_dir": manifest["source_dir"],
            "eval_holdout": manifest["eval_holdout"],
            "train_sizes": manifest["train_sizes"],
            "init_checkpoint": args.init_checkpoint,
            "epochs_per_run": args.epochs,
            "holdout_count": manifest["holdout_count"],
            "note": (
                f"Train size 15 (or max) uses all {manifest['train_pool_size']} clips "
                f"outside the {manifest['holdout_count']}-clip eval holdout."
            ),
        },
        "points": points,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {summary_path}")

    _plot_summary(summary_path, out_dir)
    print(f"Wrote {out_dir / 'transfer_curve.png'}")


if __name__ == "__main__":
    main()
