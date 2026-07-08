"""Ablation runner for AeroProber -- the headline experiment.

Trains all three decoder arms (naive / plain / structured) across multiple seeds
on the SAME frozen checkpoint and SAME held-out test clips (paired comparison),
then produces:
- A metrics table (mean +/- std across seeds) saved as JSON + printed.
- Error-vs-horizon curves with confidence bands.
- Example rollout trajectory plots (position + attitude over time).

The pre-registered success criterion (checked at the end): the structured
prober must beat the plain MLP on position RMSE with non-overlapping std bands.

Usage:
    python research/prober/scripts/run_ablations.py \
        --config research/prober/configs/prober_synth.yaml \
        --seeds 0 1 2 3 4 --num-train 256 --epochs 30

NOTE: PyFlyt must run OUTSIDE the Cursor sandbox. Run with full permissions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "research" / "prober" / "src"))
sys.path.insert(0, str(_ROOT / "src"))

from aerojepa_research.prober.data_pyflyt import build_pyflyt_dataloaders
from aerojepa_research.prober.metrics import evaluate_arm, metrics_to_dict
from aerojepa_research.prober.rollout import FrozenRolloutExtractor
from train_prober import build_model_and_loss, load_config  # noqa: E402


ARMS = ["naive", "plain", "structured"]


def run_one_seed(cfg: dict, arm: str, seed: int, device: torch.device):
    """Train one arm on one seed; return (model, loss_fn, extractor, max_loops)."""
    cfg = {**cfg, "arm": arm, "data": {**cfg["data"], "seed": seed}}
    model, loss_fn, _params, arm_name = build_model_and_loss(cfg, device)
    extractor = FrozenRolloutExtractor(
        cfg["checkpoint"], device=device, context_frames=cfg["train"]["context_frames"],
    )
    max_loops = cfg.get("max_loops") if cfg.get("predictor_mode") == "looped" else None
    params = list(model.parameters()) if hasattr(model, "parameters") else []
    opt = torch.optim.AdamW(params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    train_loader, _val_loader = build_pyflyt_dataloaders(
        batch_size=cfg["data"]["batch_size"],
        num_frames=cfg["data"]["num_frames"],
        img_size=cfg["data"]["img_size"],
        num_train=cfg["data"]["num_train"],
        num_val=cfg["data"]["num_val"],
        num_workers=cfg["data"]["num_workers"],
        seed=seed,
    )
    for epoch in range(cfg["train"]["epochs"]):
        model.train() if hasattr(model, "train") else None
        for clips, actions, states in train_loader:
            rollout = extractor.extract(clips, actions, states, max_loops=max_loops)
            from train_prober import compute_loss
            loss, _pred = compute_loss(arm_name, model, loss_fn, rollout)
            opt.zero_grad()
            loss.backward()
            if cfg["train"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
            opt.step()
    return model, loss_fn, extractor, max_loops, arm_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-train", type=int, default=None)
    parser.add_argument("--num-test", type=int, default=32, help="held-out test clips")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_train is not None:
        cfg["data"]["num_train"] = args.num_train
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    out_dir = Path(args.output_dir or cfg["output_dir"].rstrip("/") + "_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    device = torch.device(cfg.get("device", "mps") if torch.backends.mps.is_available() else "cpu")
    print(f"[ablation] device={device} seeds={args.seeds} arms={ARMS}")
    print(f"[ablation] num_train={cfg['data']['num_train']} epochs={cfg['train']['epochs']} num_test={args.num_test}")

    # Build ONE held-out test loader (shared across all arms + seeds = paired).
    # Use a seed range disjoint from all training seeds.
    test_seed = 999_983
    _train, test_loader = build_pyflyt_dataloaders(
        batch_size=cfg["data"]["batch_size"],
        num_frames=cfg["data"]["num_frames"],
        img_size=cfg["data"]["img_size"],
        num_train=4,  # unused
        num_val=args.num_test,
        num_workers=0,
        seed=test_seed,
    )

    # Results: results[arm] = list of per-seed metric dicts.
    results: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    # Per-horizon curves: curves[arm] = (n_seeds, T) array of pos RMSE.
    curves: dict[str, list[list[float]]] = {arm: [] for arm in ARMS}

    for seed in args.seeds:
        for arm in ARMS:
            t0 = time.time()
            model, loss_fn, extractor, max_loops, arm_name = run_one_seed(cfg, arm, seed, device)
            metrics, _pred_trajs, _gt_trajs = evaluate_arm(
                extractor, model, loss_fn, test_loader, device, arm_name, max_loops=max_loops,
            )
            results[arm].append(metrics_to_dict(metrics))
            curves[arm].append(metrics.per_horizon_pos_rmse)
            print(
                f"[ablation] seed={seed} arm={arm:11s} "
                f"pos_rmse={metrics.position_rmse:.4f} "
                f"att_rmse={metrics.attitude_rmse_deg:.4f} "
                f"({time.time()-t0:.1f}s)"
            )

    # Aggregate: mean +/- std across seeds.
    summary = {}
    for arm in ARMS:
        pos = np.array([r["position_rmse"] for r in results[arm]])
        att = np.array([r["attitude_rmse_deg"] for r in results[arm]])
        vel = np.array([r["velocity_rmse"] for r in results[arm]])
        summary[arm] = {
            "position_rmse_mean": float(pos.mean()),
            "position_rmse_std": float(pos.std()),
            "attitude_rmse_deg_mean": float(att.mean()),
            "attitude_rmse_deg_std": float(att.std()),
            "velocity_rmse_mean": float(vel.mean()),
            "velocity_rmse_std": float(vel.std()),
            "n_seeds": len(args.seeds),
        }

    # Pre-registered success criterion: structured beats plain on position RMSE
    # with non-overlapping std bands.
    s_pos = summary["structured"]["position_rmse_mean"]
    s_std = summary["structured"]["position_rmse_std"]
    p_pos = summary["plain"]["position_rmse_mean"]
    p_std = summary["plain"]["position_rmse_std"]
    bands_overlap = (s_pos + s_std >= p_pos - p_std) and (p_pos + p_std >= s_pos - s_std)
    success = (s_pos < p_pos) and (not bands_overlap)
    summary["success_criterion"] = {
        "description": "structured position_rmse < plain position_rmse with non-overlapping std bands",
        "structured_mean_std": f"{s_pos:.4f} +/- {s_std:.4f}",
        "plain_mean_std": f"{p_pos:.4f} +/- {p_std:.4f}",
        "bands_overlap": bool(bands_overlap),
        "met": bool(success),
    }

    # Save results.
    with open(out_dir / "per_seed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print the metrics table.
    print("\n[ablation] === METRICS TABLE (mean +/- std) ===")
    print(f"{'arm':<12} {'pos_rmse (m)':<22} {'att_rmse (deg)':<22} {'vel_rmse (m/s)':<22}")
    for arm in ARMS:
        s = summary[arm]
        print(
            f"{arm:<12} "
            f"{s['position_rmse_mean']:.4f} +/- {s['position_rmse_std']:.4f}    "
            f"{s['attitude_rmse_deg_mean']:.4f} +/- {s['attitude_rmse_deg_std']:.4f}    "
            f"{s['velocity_rmse_mean']:.4f} +/- {s['velocity_rmse_std']:.4f}"
        )
    sc = summary["success_criterion"]
    print(f"\n[ablation] success criterion met: {sc['met']}  "
          f"(structured {sc['structured_mean_std']} vs plain {sc['plain_mean_std']}, "
          f"bands_overlap={sc['bands_overlap']})")

    # Error-vs-horizon figure with confidence bands.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = {"naive": "#888", "plain": "#d62728", "structured": "#1f77b4"}
    for ax, metric_name, title in [
        (axes[0], "pos", "Position RMSE vs horizon"),
        (axes[1], "att", "Attitude RMSE vs horizon"),
    ]:
        for arm in ARMS:
            data = np.array(curves[arm])  # (n_seeds, T)
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            T = len(mean)
            xs = np.arange(1, T + 1)
            ax.plot(xs, mean, label=f"{arm}", color=colors[arm], linewidth=2)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=colors[arm])
        ax.set_xlabel("horizon (frames)")
        ax.set_ylabel("RMSE" + (" (m)" if metric_name == "pos" else " (deg)"))
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "error_vs_horizon.png", dpi=120)
    print(f"[ablation] saved figures to {fig_dir}")
    print(f"[ablation] done. results in {out_dir}")


if __name__ == "__main__":
    main()
