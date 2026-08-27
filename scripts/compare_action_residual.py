#!/usr/bin/env python
"""Compare closed-loop metrics: heuristic map vs learned residual.

Runs the same hover (and optional waypoint) closed-loop demo twice - once with
the hand-crafted AeroJEPA→PyFlyt map, once with a trained ActionResidualHead -
and writes a side-by-side JSON.

Example::

    python scripts/compare_action_residual.py \\
        --checkpoint checkpoints/action_conditioned/latest.pt \\
        --residual checkpoints/action_residual/best.pt \\
        --task hover --max-steps 120
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from aerojepa.eval import load_model
from aerojepa.sim.action_residual import load_residual_head
from aerojepa.sim.closed_loop import run_closed_loop_demo
from aerojepa.utils.device import get_device


def _summarize(results) -> dict:
    out = {}
    for name, ep in results.items():
        out[name] = {
            "steps": ep.steps,
            "total_reward": ep.total_reward,
            "altitude_mae": ep.altitude_mae,
            "xy_drift": ep.xy_drift,
            "max_xy_drift": ep.max_xy_drift,
            "survived": ep.survived,
            "final_goal_distance": ep.final_goal_distance,
            "min_goal_distance": ep.min_goal_distance,
            "reached": ep.reached,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/action_conditioned/latest.pt")
    parser.add_argument("--residual", default="checkpoints/action_residual/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--task", default="hover", choices=["hover", "waypoint", "recover"])
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--goal", type=float, nargs=3, default=None)
    parser.add_argument("--out", default="visualizations/closed_loop/residual_compare.json")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    img_size = int(cfg["data"]["img_size"])
    latent_dim = int(cfg["encoder"]["embed_dim"])
    goal = tuple(args.goal) if args.goal is not None else None

    print("=== BEFORE (heuristic only) ===")
    before = run_closed_loop_demo(
        model,
        device,
        policies=("planner", "hover", "random"),
        out_dir="visualizations/closed_loop/residual_before",
        img_size=img_size,
        max_steps=args.max_steps,
        seed=args.seed,
        task=args.task,
        goal=goal,
        num_candidates=args.num_candidates,
        residual_head=None,
    )

    print("\n=== AFTER (heuristic + learned residual) ===")
    head = load_residual_head(args.residual, device, latent_dim=latent_dim)
    print(f"Loaded residual ({head.num_params()} params) from {args.residual}")
    after = run_closed_loop_demo(
        model,
        device,
        policies=("planner", "hover", "random"),
        out_dir="visualizations/closed_loop/residual_after",
        img_size=img_size,
        max_steps=args.max_steps,
        seed=args.seed,
        task=args.task,
        goal=goal,
        num_candidates=args.num_candidates,
        residual_head=head,
    )

    report = {
        "task": args.task,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "world_checkpoint": args.checkpoint,
        "residual_checkpoint": args.residual,
        "residual_params": head.num_params(),
        "before": _summarize(before.results),
        "after": _summarize(after.results),
    }
    # Planner delta
    b = report["before"]["planner"]
    a = report["after"]["planner"]
    report["planner_delta"] = {
        "reward": a["total_reward"] - b["total_reward"],
        "altitude_mae": a["altitude_mae"] - b["altitude_mae"],
        "xy_drift": a["xy_drift"] - b["xy_drift"],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("\nPlanner comparison")
    print(f"{'':<12} {'reward':>10} {'alt_mae':>10} {'xy_end':>10}")
    print(f"{'before':<12} {b['total_reward']:10.2f} {b['altitude_mae']:10.3f} {b['xy_drift']:10.3f}")
    print(f"{'after':<12} {a['total_reward']:10.2f} {a['altitude_mae']:10.3f} {a['xy_drift']:10.3f}")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
