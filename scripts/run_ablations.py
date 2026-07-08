#!/usr/bin/env python
"""Train and evaluate a suite of AeroJEPA variants, then summarize.

Isolates the design choices that matter: feed-forward vs recurrent predictor,
loop count, and the world-model objective.

Modes:
  --mode quick   20 epochs  (~30 min on MPS) — fast signal for iteration
  --mode full   100 epochs  (~2+ hr on MPS) — publication-quality numbers

Override epoch count anytime with ``--epochs``. Skip retraining with
``--eval-only`` when checkpoints already exist under ``checkpoints/ablations/``.

Examples::

    python scripts/run_ablations.py --mode quick
    python scripts/run_ablations.py --mode full --device mps
    python scripts/run_ablations.py --epochs 5          # smoke
    python scripts/run_ablations.py --eval-only         # re-score existing ckpts

After training, render figures::

    python visualizations/compare_ablations.py
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from aerojepa.eval import load_model
from aerojepa.eval.latent_pred import latent_prediction_metrics
from aerojepa.eval.loop_metrics import loop_metrics
from aerojepa.eval.rollout import rollout_metrics
from aerojepa.masking import build_mask_collator
from aerojepa.models.looped_predictor import LoopedVideoPredictor
from aerojepa.train import build_dataloaders_from_cfg, train
from aerojepa.utils.config import load_config
from aerojepa.utils.device import get_device

MODE_EPOCHS = {"quick": 20, "full": 100}

# Each entry deep-merges onto the base config.
VARIANTS: dict[str, dict] = {
    "baseline": {
        "predictor": {"looped": False, "use_exit_gate": False},
    },
    "loops_2": {
        "predictor": {
            "looped": True,
            "max_loops": 2,
            "use_exit_gate": True,
            "norm": "rms",
            "sandwich_norm": True,
        },
    },
    "loops_3": {
        "predictor": {
            "looped": True,
            "max_loops": 3,
            "use_exit_gate": True,
            "norm": "rms",
            "sandwich_norm": True,
        },
    },
    "world_model": {
        "objective": "future",
        "masking": {"num_context_frames": 4},
        "predictor": {
            "looped": True,
            "max_loops": 3,
            "use_exit_gate": True,
            "world_model": True,
        },
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _evaluate_variant(model, cfg, device, max_batches: int) -> dict:
    _, val_loader = build_dataloaders_from_cfg(cfg)
    grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = build_mask_collator(cfg, grid, model.encoder.num_temporal)

    report: dict = {
        "objective": cfg.get("objective", "masked"),
        "latent_prediction": latent_prediction_metrics(
            model, val_loader, collator, device, cfg, max_batches
        ),
        "rollout": rollout_metrics(model, val_loader, device, cfg, max_batches=max_batches),
    }
    if isinstance(model.predictor, LoopedVideoPredictor):
        report["loop_analysis"] = loop_metrics(
            model, val_loader, collator, device, cfg, max_batches
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/aerojepa_synth_base.yaml")
    parser.add_argument(
        "--mode",
        choices=["quick", "full"],
        default=None,
        help="quick=20 epochs, full=100 epochs (overridden by --epochs)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Explicit epoch count.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=8, help="Eval batches per variant.")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training; load checkpoints/ablations/<variant>/latest.pt",
    )
    parser.add_argument("--out", default="results/ablations/summary.json")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Subset of variant names (default: all).",
    )
    args = parser.parse_args()

    if args.epochs is not None:
        epochs = args.epochs
        mode = args.mode or "custom"
    elif args.mode is not None:
        epochs = MODE_EPOCHS[args.mode]
        mode = args.mode
    else:
        epochs = MODE_EPOCHS["quick"]
        mode = "quick"

    device = get_device(args.device)
    base_cfg = load_config(args.config)
    out_path = Path(args.out)
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.variants if args.variants else list(VARIANTS.keys())
    summary: dict = {
        "meta": {
            "mode": mode,
            "epochs": epochs,
            "device": str(device),
            "config": args.config,
            "eval_only": args.eval_only,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "variants": {},
    }

    for name in names:
        if name not in VARIANTS:
            raise ValueError(f"Unknown variant {name!r}. Choose from {list(VARIANTS)}")
        print(f"\n=== variant: {name} ({epochs} epochs) ===")
        cfg = _merge(base_cfg, VARIANTS[name])
        cfg["train"]["epochs"] = epochs
        cfg["train"]["run_dir"] = f"runs/ablations/{name}"
        cfg["train"]["checkpoint_dir"] = f"checkpoints/ablations/{name}"

        ckpt_path = Path(cfg["train"]["checkpoint_dir"]) / "latest.pt"
        if args.eval_only:
            if not ckpt_path.exists():
                raise FileNotFoundError(f"--eval-only but missing {ckpt_path}")
            print(f"  loading {ckpt_path}")
        else:
            ckpt_path = train(cfg, device)

        model, loaded_cfg = load_model(str(ckpt_path), device)
        metrics = _evaluate_variant(model, loaded_cfg, device, args.max_batches)
        metrics["checkpoint"] = str(ckpt_path)
        metrics["epochs"] = epochs

        summary["variants"][name] = metrics
        lp = metrics["latent_prediction"]
        roll = metrics["rollout"]
        h4 = roll["cosine"][-1] if roll.get("cosine") else float("nan")
        print(f"  latent cosine={lp['cosine']:.4f}  rollout@h={len(roll['cosine'])}={h4:.4f}")
        if "loop_analysis" in metrics:
            loops = metrics["loop_analysis"]["per_loop_cosine"]
            print(f"  per-loop cosine={[f'{c:.3f}' for c in loops]}")

        (out_dir / f"{name}.json").write_text(json.dumps(metrics, indent=2))

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nWrote {out_path}")
    for name, m in summary["variants"].items():
        lp = m["latent_prediction"]
        print(f"  {name:14s} cosine={lp['cosine']:.4f}  ({m['objective']})")


if __name__ == "__main__":
    main()
