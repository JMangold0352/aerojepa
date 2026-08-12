#!/usr/bin/env python
"""Compare closed-loop drift: random-shooting vs gradient multi-step planning.

Runs the same task (default: ``recover``) twice with the ``planner`` policy —
once with random shooting, once with the gradient-based multi-step planner that
optimizes a differentiable cost on position, velocity, and attitude — and writes
a side-by-side drift report.

To attribute the recovery to the planner itself (rather than the reactive seek
PD blended in during recover), this uses a reduced ``--recover-seek-blend`` for
BOTH modes, so the drift difference reflects planning quality.

Example::

    python scripts/compare_planner_modes.py \\
        --checkpoint checkpoints/action_conditioned/latest.pt \\
        --task recover --max-steps 220 --recover-seek-blend 0.3

Requires PyFlyt (run outside the Cursor sandbox).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from aerojepa.eval import load_model
from aerojepa.sim.closed_loop import run_closed_loop_episode
from aerojepa.utils.device import get_device


def _drift_row(ep) -> dict:
    return {
        "planner_mode": ep.planner_mode,
        "steps": ep.steps,
        "xy_drift_final": ep.xy_drift,
        "max_xy_drift": ep.max_xy_drift,
        "post_disturb_max_xy": ep.post_disturb_max_xy,
        "recovered": ep.recovered,
        "recovery_steps": ep.recovery_steps,
        "survived": ep.survived,
        "altitude_mae": ep.altitude_mae,
        "mean_plan_cost": ep.mean_plan_cost,
        "total_reward": ep.total_reward,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/action_conditioned/latest.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--task", default="recover", choices=["recover", "waypoint", "hover"])
    parser.add_argument("--max-steps", type=int, default=220)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-candidates", type=int, default=48, help="Shooting candidates.")
    parser.add_argument("--grad-candidates", type=int, default=12)
    parser.add_argument("--grad-steps", type=int, default=20)
    parser.add_argument("--grad-lr", type=float, default=0.06)
    parser.add_argument("--grad-action-limit", type=float, default=0.2)
    parser.add_argument("--grad-vel-gain", type=float, default=1.0)
    parser.add_argument("--latent-smooth", type=float, default=0.0)
    parser.add_argument("--recover-seek-blend", type=float, default=0.30)
    parser.add_argument("--goal", type=float, nargs=3, default=None)
    parser.add_argument("--out", default="visualizations/closed_loop/planner_modes_compare.json")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    img_size = int(cfg["data"]["img_size"])
    goal = tuple(args.goal) if args.goal is not None else None

    common = dict(
        task=args.task,
        img_size=img_size,
        max_steps=args.max_steps,
        seed=args.seed,
        goal=goal,
        recover_seek_blend=args.recover_seek_blend,
        record_frames=False,
    )

    print("=== SHOOTING (random-shooting, one-step lookahead) ===")
    shoot = run_closed_loop_episode(
        model, device, policy="planner", planner_mode="shooting",
        num_candidates=args.num_candidates, **common,
    )
    print(f"  final xy drift {shoot.xy_drift:.3f}  post-disturb max {shoot.post_disturb_max_xy}")

    print("=== GRADIENT (multi-step, differentiable pos/vel/att cost) ===")
    grad = run_closed_loop_episode(
        model, device, policy="planner", planner_mode="gradient",
        grad_candidates=args.grad_candidates, grad_steps=args.grad_steps,
        grad_lr=args.grad_lr, grad_action_limit=args.grad_action_limit,
        grad_vel_gain=args.grad_vel_gain, latent_smooth=args.latent_smooth, **common,
    )
    print(f"  final xy drift {grad.xy_drift:.3f}  post-disturb max {grad.post_disturb_max_xy}")

    report = {
        "task": args.task,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "recover_seek_blend": args.recover_seek_blend,
        "world_checkpoint": args.checkpoint,
        "grad_steps": args.grad_steps,
        "grad_candidates": args.grad_candidates,
        "latent_smooth": args.latent_smooth,
        "shooting": _drift_row(shoot),
        "gradient": _drift_row(grad),
    }
    if args.task == "recover" and shoot.post_disturb_max_xy and grad.post_disturb_max_xy:
        report["improvement"] = {
            "final_xy_drift": shoot.xy_drift - grad.xy_drift,
            "post_disturb_max_xy": shoot.post_disturb_max_xy - grad.post_disturb_max_xy,
            "final_xy_drift_pct": 100.0 * (shoot.xy_drift - grad.xy_drift) / max(shoot.xy_drift, 1e-6),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("\nDrift comparison (planner policy)")
    print(f"{'mode':<12} {'final_xy':>10} {'max_xy':>10} {'postkick':>10} {'recovered':>10}")
    for name, ep in (("shooting", shoot), ("gradient", grad)):
        pk = "—" if ep.post_disturb_max_xy is None else f"{ep.post_disturb_max_xy:.3f}"
        rec = "—" if ep.recovered is None else str(ep.recovered)
        print(f"{name:<12} {ep.xy_drift:10.3f} {ep.max_xy_drift:10.3f} {pk:>10} {rec:>10}")
    print(f"\nreport → {out}")


if __name__ == "__main__":
    main()
