#!/usr/bin/env python
"""Closed-loop PyFlyt demo: camera → AeroJEPA planner → physics.

Runs the latent planner inside ``PyFlyt/QuadX-Hover-v4`` and compares it to
baselines. Writes metrics JSON, a trajectory plot, and per-policy GIFs under
``visualizations/closed_loop/``.

Examples::

    # Default v1 stack (Wilds action + residual):
    python scripts/run_closed_loop_demo.py \\
        --checkpoint checkpoints/action_conditioned_wilds/latest.pt \\
        --residual-checkpoint checkpoints/action_residual_wilds/best.pt \\
        --planner gradient --latent-smooth 0.05 --task hover

    # Waypoint:
    python scripts/run_closed_loop_demo.py \\
        --checkpoint checkpoints/action_conditioned_wilds/latest.pt \\
        --residual-checkpoint checkpoints/action_residual_wilds/best.pt \\
        --planner gradient --latent-smooth 0.05 \\
        --task waypoint --goal 0.6 0.0 0.0 --max-steps 200

    # Disturbance recovery:
    python scripts/run_closed_loop_demo.py \\
        --checkpoint checkpoints/action_conditioned_wilds/latest.pt \\
        --residual-checkpoint checkpoints/action_residual_wilds/best.pt \\
        --planner gradient --latent-smooth 0.05 \\
        --task recover --max-steps 220

``*_wilds_v2`` checkpoints are worse on protocol-B / soft turn and are not defaults.

Requires optional deps: ``pip install PyFlyt gymnasium``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import torch

from aerojepa.eval import load_model
from aerojepa.models.jepa import AeroJEPA
from aerojepa.sim.closed_loop import (
    DEFAULT_AGGRESSIVE_LEG1,
    DEFAULT_AGGRESSIVE_LEG2,
    DEFAULT_WAYPOINT_GOAL,
    DEFAULT_WIND_MPS,
    DEFAULT_WIND_ONSET,
    run_closed_loop_demo,
)
from aerojepa.utils.config import load_config
from aerojepa.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/action_conditioned_wilds/latest.pt",
        help="World-model checkpoint (default: v1 Wilds action-conditioned).",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--task",
        default="hover",
        choices=["hover", "waypoint", "smoothness", "recover", "wind_gust", "aggressive_turn"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--num-candidates", type=int, default=48)
    parser.add_argument(
        "--action-scale",
        type=float,
        default=None,
        help="Candidate action noise scale (default: 0.04 hover / 0.10 waypoint|recover / 0.14 aggressive).",
    )
    parser.add_argument(
        "--replan-every",
        type=int,
        default=4,
        help=(
            "Replan period in steps. If planning-forward p95 exceeds the 25 ms "
            "budget (results/inference_latency.json), increase this or shorten "
            "--horizon rather than growing the world model."
        ),
    )
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument(
        "--goal",
        type=float,
        nargs=3,
        default=None,
        metavar=("DX", "DY", "DZ"),
        help="Waypoint displacement from start (defaults to 0.6 0 0).",
    )
    parser.add_argument(
        "--reach-threshold",
        type=float,
        default=0.25,
        help="Meters; mark episode reached if min distance ≤ this.",
    )
    parser.add_argument("--disturb-at", type=int, default=30, help="Step when the kick starts (recover).")
    parser.add_argument("--disturb-steps", type=int, default=18, help="Kick duration in steps (recover).")
    parser.add_argument("--damp-steps", type=int, default=15, help="Zero-rate damp after kick (recover).")
    parser.add_argument(
        "--recover-xy-threshold",
        type=float,
        default=0.40,
        help="XY meters from start to count as recovered.",
    )
    parser.add_argument(
        "--wind-mps",
        type=float,
        default=DEFAULT_WIND_MPS,
        help="Constant lateral wind speed for wind_gust (moderate default 2 m/s).",
    )
    parser.add_argument(
        "--wind-onset",
        type=int,
        default=DEFAULT_WIND_ONSET,
        help="Steps of calm settle before wind engages (wind_gust).",
    )
    parser.add_argument(
        "--no-assist-altitude",
        action="store_true",
        help="Disable the altitude PD overlay (harder; drones may crash sooner).",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        choices=["planner", "hover", "random", "inert", "seek"],
        help="Default depends on task (hover / waypoint|recover|wind|turn).",
    )
    parser.add_argument(
        "--residual-checkpoint",
        default="checkpoints/action_residual_wilds/best.pt",
        help="ActionResidualHead checkpoint (default: v1 Wilds multi-stress residual).",
    )
    parser.add_argument(
        "--planner",
        choices=["shooting", "gradient"],
        default="gradient",
        help="Planner mode (default: gradient for the v1 full stack).",
    )
    parser.add_argument("--grad-steps", type=int, default=20, help="Gradient planner iterations.")
    parser.add_argument("--grad-lr", type=float, default=0.06, help="Gradient planner learning rate.")
    parser.add_argument(
        "--grad-candidates", type=int, default=12, help="Parallel plans refined by the gradient planner."
    )
    parser.add_argument(
        "--grad-action-limit",
        type=float,
        default=0.2,
        help="Per-step |action| clamp for the gradient planner (keeps plans gentle).",
    )
    parser.add_argument(
        "--grad-vel-gain",
        type=float,
        default=1.0,
        help="Momentum gain: how strongly the gradient planner brakes current velocity.",
    )
    parser.add_argument(
        "--latent-smooth",
        type=float,
        default=0.05,
        help="Weight on the world-model latent-smoothness term (>0 = plan in latent space).",
    )
    parser.add_argument(
        "--latent-refine-steps",
        type=int,
        default=8,
        help="Last N gradient steps that use world-model latents when --latent-smooth > 0.",
    )
    parser.add_argument(
        "--recover-seek-blend",
        type=float,
        default=0.70,
        help="Fraction of the reactive seek PD blended in during recover (lower = more planner).",
    )
    parser.add_argument(
        "--strict-realtime",
        action="store_true",
        help="Exit non-zero if mean loop_ms exceeds the agent_hz budget (off by default; laptops miss 40 Hz).",
    )
    parser.add_argument("--out-dir", default="visualizations/closed_loop")
    args = parser.parse_args()

    device = get_device(args.device)
    if args.checkpoint and Path(args.checkpoint).exists():
        model, cfg = load_model(args.checkpoint, device)
        print(f"Loaded {args.checkpoint}")
    else:
        cfg = load_config("configs/smoke_test.yaml")
        model = AeroJEPA.from_config(cfg).to(device).eval()
        print("No checkpoint given -> UNTRAINED smoke model (pipeline demo only).")

    img_size = int(cfg["data"]["img_size"])
    latent_dim = int(cfg["encoder"]["embed_dim"])
    residual_head = None
    if args.residual_checkpoint and Path(args.residual_checkpoint).exists():
        from aerojepa.sim.action_residual import load_residual_head

        residual_head = load_residual_head(
            args.residual_checkpoint, device, latent_dim=latent_dim
        )
        print(
            f"Loaded residual ({residual_head.num_params()} params) "
            f"from {args.residual_checkpoint}"
        )
    elif args.residual_checkpoint:
        print(f"Residual checkpoint not found ({args.residual_checkpoint}); heuristic map only.")

    conditioned = model.predictor_is_action_conditioned()
    if args.goal is not None:
        goal = tuple(args.goal)
    elif args.task == "waypoint":
        goal = DEFAULT_WAYPOINT_GOAL
    elif args.task == "aggressive_turn":
        goal = DEFAULT_AGGRESSIVE_LEG2
    else:
        goal = None
    action_scale = args.action_scale
    if action_scale is None:
        if args.task == "aggressive_turn":
            action_scale = 0.14
        elif args.task in ("waypoint", "recover"):
            action_scale = 0.10
        else:
            action_scale = 0.04

    policies = tuple(args.policies) if args.policies is not None else None
    print(
        f"Action-conditioned: {conditioned}  |  task: {args.task}  |  "
        f"img_size: {img_size}  |  goal: {goal}"
    )
    if args.task == "wind_gust":
        print(f"  wind: {args.wind_mps} m/s from step {args.wind_onset} (heuristic map unchanged)")
    if args.task == "aggressive_turn":
        print(f"  L-turn: {DEFAULT_AGGRESSIVE_LEG1} → {DEFAULT_AGGRESSIVE_LEG2}")
    if not conditioned and args.task in (
        "hover",
        "waypoint",
        "recover",
        "wind_gust",
        "aggressive_turn",
    ):
        print("  (plain world model: candidates scored by kinematic cost, not latents.)")

    out = run_closed_loop_demo(
        model,
        device,
        policies=policies,  # type: ignore[arg-type]
        out_dir=args.out_dir,
        img_size=img_size,
        max_steps=args.max_steps,
        seed=args.seed,
        task=args.task,
        goal=goal,
        assist_altitude=not args.no_assist_altitude,
        reach_threshold=args.reach_threshold,
        num_candidates=args.num_candidates,
        action_scale=action_scale,
        replan_every=args.replan_every,
        horizon=args.horizon,
        context_frames=args.context_frames,
        disturb_at=args.disturb_at,
        disturb_steps=args.disturb_steps,
        recover_xy_threshold=args.recover_xy_threshold,
        damp_steps=args.damp_steps,
        residual_head=residual_head,
        planner_mode=args.planner,
        grad_steps=args.grad_steps,
        grad_lr=args.grad_lr,
        grad_candidates=args.grad_candidates,
        grad_action_limit=args.grad_action_limit,
        grad_vel_gain=args.grad_vel_gain,
        latent_smooth=args.latent_smooth,
        latent_refine_steps=args.latent_refine_steps,
        recover_seek_blend=args.recover_seek_blend,
        wind_mps=args.wind_mps,
        wind_onset=args.wind_onset,
        aggressive_leg1=DEFAULT_AGGRESSIVE_LEG1,
        aggressive_leg2=DEFAULT_AGGRESSIVE_LEG2,
    )

    print("\nClosed-loop results")
    if args.task in ("waypoint", "aggressive_turn"):
        print(
            f"{'policy':<10} {'steps':>6} {'reward':>10} {'min_dist':>10} "
            f"{'legs':>8} {'fail':>16} {'ok':>5}"
        )
        for name, ep in out.results.items():
            legs = "-"
            if ep.waypoints_total:
                legs = f"{ep.waypoints_reached}/{ep.waypoints_total}"
            elif ep.reached is not None:
                legs = "yes" if ep.reached else "no"
            print(
                f"{name:<10} {ep.steps:6d} {ep.total_reward:10.2f} "
                f"{(ep.min_goal_distance or 0):10.3f} {legs:>8} "
                f"{(ep.failure_mode or '-'):>16} {str(ep.survived):>5}"
            )
            if ep.failure_detail:
                print(f"           └─ {ep.failure_detail}")
    elif args.task == "recover":
        print(
            f"{'policy':<10} {'steps':>6} {'reward':>10} {'rec_steps':>10} "
            f"{'xy_end':>10} {'fail':>16} {'ok':>5}"
        )
        for name, ep in out.results.items():
            rec = "-" if ep.recovery_steps is None else str(ep.recovery_steps)
            print(
                f"{name:<10} {ep.steps:6d} {ep.total_reward:10.2f} "
                f"{rec:>10} {ep.xy_drift:10.3f} "
                f"{(ep.failure_mode or '-'):>16} {str(ep.survived):>5}"
            )
            if ep.failure_detail:
                print(f"           └─ {ep.failure_detail}")
    elif args.task == "wind_gust":
        print(
            f"{'policy':<10} {'steps':>6} {'reward':>10} {'alt_mae':>10} "
            f"{'max_xy':>10} {'fail':>16} {'ok':>5}"
        )
        for name, ep in out.results.items():
            print(
                f"{name:<10} {ep.steps:6d} {ep.total_reward:10.2f} "
                f"{ep.altitude_mae:10.3f} {ep.max_xy_drift:10.3f} "
                f"{(ep.failure_mode or '-'):>16} {str(ep.survived):>5}"
            )
            if ep.failure_detail:
                print(f"           └─ {ep.failure_detail}")
    else:
        print(
            f"{'policy':<10} {'steps':>6} {'reward':>10} {'alt_mae':>10} "
            f"{'xy_end':>10} {'fail':>16} {'ok':>5}"
        )
        for name, ep in out.results.items():
            print(
                f"{name:<10} {ep.steps:6d} {ep.total_reward:10.2f} "
                f"{ep.altitude_mae:10.3f} {ep.xy_drift:10.3f} "
                f"{(ep.failure_mode or '-'):>16} {str(ep.survived):>5}"
            )
            if ep.failure_detail:
                print(f"           └─ {ep.failure_detail}")
    print("\nTiming / watchdog (research hold, not a flight watchdog)")
    for name, ep in out.results.items():
        mean_ms = ep.mean_loop_ms if ep.mean_loop_ms is not None else float("nan")
        p95_ms = ep.p95_loop_ms if ep.p95_loop_ms is not None else float("nan")
        print(
            f"  {name:<10} mean_loop_ms={mean_ms:7.2f}  p95_loop_ms={p95_ms:7.2f}  "
            f"budget_ms={ep.budget_ms:.1f}  watchdog_holds={ep.watchdog_holds}"
        )
    print(f"\nmetrics -> {out.metrics_path}")
    print(f"plot    -> {out.plot_path}")
    for name, path in out.gif_paths.items():
        print(f"gif[{name}] -> {path}")

    if args.strict_realtime:
        for name, ep in out.results.items():
            if ep.mean_loop_ms is not None and ep.mean_loop_ms > ep.budget_ms:
                raise SystemExit(
                    f"--strict-realtime: {name} mean_loop_ms={ep.mean_loop_ms:.2f} "
                    f"> budget_ms={ep.budget_ms:.1f}"
                )


if __name__ == "__main__":
    main()
