#!/usr/bin/env python
"""Full-stack closed-loop comparison: baseline vs gradient+residual (+ latent).

Runs a compact task suite with the heuristic map alone, then with residual +
gradient planner (+ optional latent refine). Writes a comparison table.
Supports multiple seeds; the report has per-seed rows plus an aggregate block.

Example::

    python scripts/compare_full_stack.py \\
        --checkpoint checkpoints/action_conditioned/latest.pt \\
        --residual checkpoints/action_residual_wind/best.pt \\
        --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from aerojepa.eval import load_model
from aerojepa.sim.action_residual import load_residual_head
from aerojepa.sim.closed_loop import (
    DEFAULT_AGGRESSIVE_LEG1,
    DEFAULT_AGGRESSIVE_LEG1_HARD,
    DEFAULT_AGGRESSIVE_LEG2,
    DEFAULT_AGGRESSIVE_LEG2_HARD,
    run_closed_loop_episode,
)
from aerojepa.utils.device import get_device


def _row(ep) -> dict:
    return {
        "planner_mode": ep.planner_mode,
        "steps": ep.steps,
        "survived": ep.survived,
        "total_reward": round(ep.total_reward, 2),
        "altitude_mae": round(ep.altitude_mae, 3),
        "xy_drift": round(ep.xy_drift, 3),
        "max_xy_drift": round(ep.max_xy_drift, 3),
        "reached": ep.reached,
        "recovered": ep.recovered,
        "recovery_steps": ep.recovery_steps,
        "waypoints_reached": ep.waypoints_reached,
        "waypoints_total": ep.waypoints_total,
        "failure_mode": ep.failure_mode,
        "failure_detail": ep.failure_detail,
        "mean_plan_cost": None if ep.mean_plan_cost is None else round(ep.mean_plan_cost, 4),
    }


def _success(label: str, row: dict) -> bool:
    if label.startswith("aggressive_turn"):
        return row["failure_mode"] == "ok"
    if label == "recover":
        return bool(row["recovered"]) and row["failure_mode"] == "ok"
    return row["failure_mode"] == "ok"


def _aggregate(label: str, rows: list[dict]) -> dict:
    agg: dict = {
        "n_seeds": len(rows),
        "success_rate": round(sum(_success(label, r) for r in rows) / len(rows), 3),
        "mean_xy_drift": round(float(np.mean([r["xy_drift"] for r in rows])), 3),
        "mean_max_xy_drift": round(float(np.mean([r["max_xy_drift"] for r in rows])), 3),
    }
    if label.startswith("aggressive_turn"):
        agg["mean_legs"] = round(
            float(np.mean([r["waypoints_reached"] or 0 for r in rows])), 2
        )
    if label == "recover":
        rec = [r["recovery_steps"] for r in rows if r["recovery_steps"] is not None]
        agg["mean_recovery_steps"] = round(float(np.mean(rec)), 1) if rec else None
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/action_conditioned/latest.pt")
    parser.add_argument("--residual", default="checkpoints/action_residual_wind/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--wind-mps", type=float, default=2.0)
    parser.add_argument("--latent-smooth", type=float, default=0.05)
    parser.add_argument("--latent-refine-steps", type=int, default=8)
    parser.add_argument(
        "--include-hard-turn",
        action="store_true",
        help="Also run the hard 0.8 m L-course (full stack only baseline comparison).",
    )
    parser.add_argument("--out", default="visualizations/closed_loop/full_stack_compare.json")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    img_size = int(cfg["data"]["img_size"])
    latent_dim = int(cfg["encoder"]["embed_dim"])
    head = load_residual_head(args.residual, device, latent_dim=latent_dim)
    print(f"Loaded residual ({head.num_params()} params) from {args.residual}")

    tasks = [
        ("wind_gust", "wind_gust", dict(max_steps=200, wind_mps=args.wind_mps, wind_onset=40)),
        (
            "aggressive_turn",
            "aggressive_turn",
            dict(
                max_steps=260,
                action_scale=0.10,
                aggressive_leg1=DEFAULT_AGGRESSIVE_LEG1,
                aggressive_leg2=DEFAULT_AGGRESSIVE_LEG2,
                horizon=4,
            ),
        ),
        ("recover", "recover", dict(max_steps=220, recover_seek_blend=0.30)),
        ("hover", "hover", dict(max_steps=120)),
    ]
    if args.include_hard_turn:
        tasks.append(
            (
                "aggressive_turn_hard",
                "aggressive_turn",
                dict(
                    max_steps=320,
                    action_scale=0.10,
                    aggressive_leg1=DEFAULT_AGGRESSIVE_LEG1_HARD,
                    aggressive_leg2=DEFAULT_AGGRESSIVE_LEG2_HARD,
                    horizon=4,
                ),
            )
        )

    report: dict = {
        "world_checkpoint": args.checkpoint,
        "residual_checkpoint": args.residual,
        "residual_params": head.num_params(),
        "seeds": args.seeds,
        "latent_smooth": args.latent_smooth,
        "tasks": {},
    }

    for label, task, kw in tasks:
        base_rows: list[dict] = []
        full_rows: list[dict] = []
        hover_rows: list[dict] = []
        for seed in args.seeds:
            print(f"\n=== {label} seed={seed}: baseline (shooting, no residual) ===")
            base = run_closed_loop_episode(
                model,
                device,
                policy="planner",
                task=task,
                img_size=img_size,
                seed=seed,
                planner_mode="shooting",
                residual_head=None,
                record_frames=False,
                **kw,
            )
            print(f"  → {base.failure_mode}  max_xy={base.max_xy_drift:.3f}")
            base_rows.append(_row(base))

            print(f"=== {label} seed={seed}: full stack (gradient + residual + latent) ===")
            full = run_closed_loop_episode(
                model,
                device,
                policy="planner",
                task=task,
                img_size=img_size,
                seed=seed,
                planner_mode="gradient",
                residual_head=head,
                latent_smooth=args.latent_smooth,
                latent_refine_steps=args.latent_refine_steps,
                grad_steps=20,
                grad_candidates=12,
                record_frames=False,
                **kw,
            )
            print(f"  → {full.failure_mode}  max_xy={full.max_xy_drift:.3f}")
            full_rows.append(_row(full))

            if task == "wind_gust":
                hover = run_closed_loop_episode(
                    None,
                    device,
                    policy="hover",
                    task=task,
                    img_size=img_size,
                    seed=seed,
                    record_frames=False,
                    **kw,
                )
                hover_rows.append(_row(hover))

        report["tasks"][label] = {
            "per_seed": {
                "baseline": base_rows,
                "full_stack": full_rows,
                "hover": hover_rows or None,
            },
            "aggregate": {
                "baseline": _aggregate(label, base_rows),
                "full_stack": _aggregate(label, full_rows),
                "hover": _aggregate(label, hover_rows) if hover_rows else None,
            },
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    n = len(args.seeds)
    print(f"\n=== Full-stack summary ({n} seed{'s' if n > 1 else ''}) ===")
    print(f"{'task':<22} {'base ok':>8} {'full ok':>8} {'base max_xy':>12} {'full max_xy':>12}")
    for label, block in report["tasks"].items():
        b, f = block["aggregate"]["baseline"], block["aggregate"]["full_stack"]
        print(
            f"{label:<22} {b['success_rate']:>8.0%} {f['success_rate']:>8.0%} "
            f"{b['mean_max_xy_drift']:>12.2f} {f['mean_max_xy_drift']:>12.2f}"
        )
        h = block["aggregate"].get("hover")
        if h:
            print(f"{'  (hover ref)':<22} {h['success_rate']:>8.0%} {'':>8} {h['mean_max_xy_drift']:>12.2f}")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
