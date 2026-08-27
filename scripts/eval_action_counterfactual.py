#!/usr/bin/env python
"""Action-counterfactual eval for action-conditioned AeroJEPA.

Scores future-latent cosine + smooth-L1 under true / zero / shuffled actions.
Optional 2D action-energy heatmap (mean L1 vs GT future) over two DoF.

Success: shuffled/zero clearly worse than true if the model uses actions.
If not, report that plainly (F2). No retraining.

Example::

    python scripts/eval_action_counterfactual.py \\
        --checkpoint checkpoints/action_conditioned_wilds/latest.pt \\
        --data-dir data/flights_128 --max-batches 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from aerojepa.data.telemetry import ACTION_COLUMNS
from aerojepa.eval import load_model
from aerojepa.masking import build_mask_collator
from aerojepa.train import _prep_actions, build_video_dataloaders, stack_indices
from aerojepa.utils.device import get_device


@torch.no_grad()
def _score_batch(model, clips, ctx, tgt, acts) -> tuple[float, float]:
    out = model(clips, ctx, tgt, actions=acts)
    cos = float(F.cosine_similarity(out["pred_repr"], out["target_repr"], dim=-1).mean())
    l1 = float(F.smooth_l1_loss(out["pred_repr"], out["target_repr"]))
    return cos, l1


@torch.no_grad()
def counterfactual_metrics(
    model,
    loader,
    collator,
    device: torch.device,
    cfg: dict,
    max_batches: int = 8,
) -> dict:
    """Aggregate cosine/L1 under true, zero, and batch-shuffled actions."""
    model.eval()
    num_temporal = model.encoder.num_temporal
    buckets = {
        "true": {"cos": 0.0, "l1": 0.0},
        "zero": {"cos": 0.0, "l1": 0.0},
        "shuffle": {"cos": 0.0, "l1": 0.0},
    }
    n = 0
    for i, (clips, actions) in enumerate(loader):
        if i >= max_batches:
            break
        clips = clips.to(device)
        masks = collator(clips.shape[0])
        ctx = stack_indices(masks.context_indices, device)
        tgt = stack_indices(masks.target_indices, device)
        acts = _prep_actions(actions, num_temporal, device, cfg)

        zero = torch.zeros_like(acts)
        # Permute across the batch so each clip sees another clip's actions.
        perm = torch.randperm(acts.shape[0], device=device)
        shuffled = acts[perm]

        for name, a in (("true", acts), ("zero", zero), ("shuffle", shuffled)):
            cos, l1 = _score_batch(model, clips, ctx, tgt, a)
            buckets[name]["cos"] += cos
            buckets[name]["l1"] += l1
        n += 1

    n = max(1, n)
    out = {}
    for name, b in buckets.items():
        out[name] = {"cosine": b["cos"] / n, "smooth_l1": b["l1"] / n, "n_batches": n}
    return out


@torch.no_grad()
def action_energy_heatmap(
    model,
    loader,
    collator,
    device: torch.device,
    cfg: dict,
    dim_i: int = 0,
    dim_j: int = 1,
    grid: int = 9,
    max_batches: int = 4,
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> dict:
    """Mean latent L1 vs GT future while sweeping two action dims (others = true)."""
    model.eval()
    num_temporal = model.encoder.num_temporal
    lo, hi = value_range
    vals = np.linspace(lo, hi, grid)
    heat = np.zeros((grid, grid), dtype=np.float64)
    counts = 0
    true_means: list[tuple[float, float]] = []

    for i, (clips, actions) in enumerate(loader):
        if i >= max_batches:
            break
        clips = clips.to(device)
        masks = collator(clips.shape[0])
        ctx = stack_indices(masks.context_indices, device)
        tgt = stack_indices(masks.target_indices, device)
        acts = _prep_actions(actions, num_temporal, device, cfg)
        true_means.append(
            (
                float(acts[..., dim_i].mean()),
                float(acts[..., dim_j].mean()),
            )
        )
        for ii, vi in enumerate(vals):
            for jj, vj in enumerate(vals):
                a = acts.clone()
                a[..., dim_i] = vi
                a[..., dim_j] = vj
                _, l1 = _score_batch(model, clips, ctx, tgt, a)
                heat[ii, jj] += l1
        counts += 1

    counts = max(1, counts)
    heat /= counts
    return {
        "dim_i": dim_i,
        "dim_j": dim_j,
        "dim_i_name": ACTION_COLUMNS[dim_i],
        "dim_j_name": ACTION_COLUMNS[dim_j],
        "values": vals.tolist(),
        "heatmap_l1": heat.tolist(),
        "true_action_means": true_means,
    }


def _verdict(report: dict) -> str:
    t = report["counterfactual"]["true"]
    z = report["counterfactual"]["zero"]
    s = report["counterfactual"]["shuffle"]
    # Model "uses" actions if true has higher cosine OR lower L1 than both baselines.
    uses = (t["cosine"] > z["cosine"] + 0.005 and t["cosine"] > s["cosine"] + 0.005) or (
        t["smooth_l1"] < z["smooth_l1"] * 0.98 and t["smooth_l1"] < s["smooth_l1"] * 0.98
    )
    if uses:
        return (
            "PASS: true actions beat zero and shuffle on latent metrics - "
            "predictor is action-sensitive."
        )
    return (
        "FAIL (F2): true actions do not clearly beat zero/shuffle. "
        "Action-conditioning may be ignored; do not claim a causal world model yet."
    )


def _plot(report: dict, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    names = ["true", "zero", "shuffle"]
    cos = [report["counterfactual"][n]["cosine"] for n in names]
    l1 = [report["counterfactual"][n]["smooth_l1"] for n in names]
    axes[0].bar(names, cos, color=["#2a9d8f", "#e76f51", "#264653"])
    axes[0].set_ylabel("latent cosine (↑)")
    axes[0].set_title("Counterfactual cosine")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(names, l1, color=["#2a9d8f", "#e76f51", "#264653"])
    axes[1].set_ylabel("smooth-L1 (↓)")
    axes[1].set_title("Counterfactual L1")

    fig.suptitle(
        f"Action counterfactuals - {Path(report['checkpoint']).parent.name}\n"
        f"{report['verdict'][:80]}…",
        fontsize=10,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    heat = report.get("heatmap")
    if not heat:
        return
    hm = np.asarray(heat["heatmap_l1"])
    fig2, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(hm, origin="lower", cmap="magma")
    ax.set_xlabel(heat["dim_j_name"])
    ax.set_ylabel(heat["dim_i_name"])
    ax.set_title("Action energy (mean latent L1 vs GT)")
    ticks = np.linspace(0, len(heat["values"]) - 1, 5).astype(int)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([f"{heat['values'][t]:.2f}" for t in ticks])
    ax.set_yticklabels([f"{heat['values'][t]:.2f}" for t in ticks])
    for mx, my in heat["true_action_means"]:
        # Map continuous action means onto grid indices (rows=dim_i, cols=dim_j).
        vals = np.asarray(heat["values"])
        ii = int(np.argmin(np.abs(vals - mx)))
        jj = int(np.argmin(np.abs(vals - my)))
        ax.plot(jj, ii, "c+", markersize=12, markeredgewidth=2)
    fig2.colorbar(im, ax=ax, fraction=0.046, label="smooth-L1")
    heat_png = out_png.with_name(out_png.stem + "_heatmap.png")
    fig2.tight_layout()
    fig2.savefig(heat_png, dpi=150)
    plt.close(fig2)
    report["heatmap_figure"] = str(heat_png)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/action_conditioned_wilds/latest.pt",
    )
    parser.add_argument(
        "--compare-checkpoint",
        default="checkpoints/real_finetune_fast/latest.pt",
        help="Optional unconditioned baseline (skipped if missing).",
    )
    parser.add_argument("--data-dir", default="data/flights_128")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--heatmap-grid", type=int, default=7)
    parser.add_argument("--heatmap-batches", type=int, default=3)
    parser.add_argument(
        "--out",
        default="results/action_counterfactual.json",
    )
    parser.add_argument(
        "--figure",
        default="visualizations/figures/action_counterfactual.png",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    if not model.predictor_is_action_conditioned():
        raise SystemExit(
            f"{args.checkpoint} is not action-conditioned; pick an AC checkpoint."
        )

    data_cfg = dict(cfg["data"])
    data_cfg["source"] = "video"
    data_cfg["data_dir"] = args.data_dir
    # Use sliding windows over *all* videos as the eval pool (diagnostic, not
    # the Protocol-B train/val split). Caps via --max-batches.
    data_cfg["window_mode"] = "sliding"
    data_cfg["val_fraction"] = 1.0  # every video in the "val" loader
    data_cfg["batch_size"] = min(int(data_cfg.get("batch_size", 4)), 4)
    data_cfg["num_workers"] = 0
    _, val_loader = build_video_dataloaders(data_cfg)

    grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = build_mask_collator(cfg, grid, model.encoder.num_temporal)

    cf = counterfactual_metrics(
        model, val_loader, collator, device, cfg, max_batches=args.max_batches
    )
    heat = action_energy_heatmap(
        model,
        val_loader,
        collator,
        device,
        cfg,
        dim_i=0,
        dim_j=1,
        grid=args.heatmap_grid,
        max_batches=args.heatmap_batches,
    )

    report: dict = {
        "checkpoint": args.checkpoint,
        "data_dir": args.data_dir,
        "protocol": "action_counterfactual_v1",
        "counterfactual": cf,
        "heatmap": heat,
    }
    report["verdict"] = _verdict(report)

    # Unconditioned baseline: same clips, actions ignored (forward with None).
    cmp_path = Path(args.compare_checkpoint)
    if cmp_path.is_file():
        umodel, ucfg = load_model(cmp_path, device)
        ugrid = ucfg["data"]["img_size"] // ucfg["data"]["patch_size"]
        ucoll = build_mask_collator(ucfg, ugrid, umodel.encoder.num_temporal)
        # Rebuild loader with uncond cfg frame settings if needed.
        udata = dict(ucfg["data"])
        udata["source"] = "video"
        udata["data_dir"] = args.data_dir
        udata["batch_size"] = min(int(udata.get("batch_size", 4)), 4)
        udata["num_workers"] = 0
        _, uval = build_video_dataloaders(udata)
        n = 0
        cos_sum = l1_sum = 0.0
        for i, (clips, actions) in enumerate(uval):
            if i >= args.max_batches:
                break
            clips = clips.to(device)
            masks = ucoll(clips.shape[0])
            ctx = stack_indices(masks.context_indices, device)
            tgt = stack_indices(masks.target_indices, device)
            out = umodel(clips, ctx, tgt, actions=None)
            cos_sum += float(
                F.cosine_similarity(out["pred_repr"], out["target_repr"], dim=-1)
                .mean()
                .detach()
            )
            l1_sum += float(F.smooth_l1_loss(out["pred_repr"], out["target_repr"]).detach())
            n += 1
        n = max(1, n)
        report["unconditioned_compare"] = {
            "checkpoint": str(cmp_path),
            "cosine": cos_sum / n,
            "smooth_l1": l1_sum / n,
            "note": "Uncond model has no action input; not a counterfactual arm.",
        }

    print("=== action counterfactuals ===")
    for name in ("true", "zero", "shuffle"):
        r = cf[name]
        print(f"  {name:8s}  cosine={r['cosine']:.4f}  l1={r['smooth_l1']:.4f}")
    print(report["verdict"])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path = Path(args.figure)
    _plot(report, fig_path)
    report["figure"] = str(fig_path)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
