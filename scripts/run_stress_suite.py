#!/usr/bin/env python
"""Run the closed-loop stress suite (wind gust + aggressive turn) and summarize breaks.

Keeps the same heuristic AeroJEPA→PyFlyt action map for a fair comparison across
policies. Starts at moderate wind (2 m/s) and an L-shaped 90° turn.

Writes per-task metrics/GIFs under ``visualizations/closed_loop/stress/`` plus a
breaking-points summary JSON.

Example::

    python scripts/run_stress_suite.py \\
        --checkpoint checkpoints/action_conditioned/latest.pt \\
        --wind-mps 2.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from aerojepa.eval import load_model
from aerojepa.sim.closed_loop import (
    DEFAULT_AGGRESSIVE_LEG1,
    DEFAULT_AGGRESSIVE_LEG2,
    DEFAULT_WIND_MPS,
    DEFAULT_WIND_ONSET,
    run_closed_loop_demo,
)
from aerojepa.utils.device import get_device


def _policy_row(ep) -> dict:
    return {
        "steps": ep.steps,
        "survived": ep.survived,
        "total_reward": round(ep.total_reward, 2),
        "altitude_mae": round(ep.altitude_mae, 3),
        "xy_drift": round(ep.xy_drift, 3),
        "max_xy_drift": round(ep.max_xy_drift, 3),
        "reached": ep.reached,
        "waypoints_reached": ep.waypoints_reached,
        "waypoints_total": ep.waypoints_total,
        "failure_mode": ep.failure_mode,
        "failure_detail": ep.failure_detail,
        "wind_mps": ep.wind_mps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/action_conditioned/latest.pt")
    parser.add_argument(
        "--residual-checkpoint",
        default=None,
        help="Optional ActionResidualHead (enables full-stack stress when set).",
    )
    parser.add_argument(
        "--planner",
        default="shooting",
        choices=["shooting", "gradient"],
        help="Planner mode for the AeroJEPA policy (use gradient with residual).",
    )
    parser.add_argument("--latent-smooth", type=float, default=0.0)
    parser.add_argument("--latent-refine-steps", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wind-mps", type=float, default=DEFAULT_WIND_MPS)
    parser.add_argument("--wind-onset", type=int, default=DEFAULT_WIND_ONSET)
    parser.add_argument("--wind-steps", type=int, default=200)
    parser.add_argument("--turn-steps", type=int, default=260)
    parser.add_argument("--out-dir", default="visualizations/closed_loop/stress")
    args = parser.parse_args()

    device = get_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    img_size = int(cfg["data"]["img_size"])
    latent_dim = int(cfg["encoder"]["embed_dim"])
    residual_head = None
    if args.residual_checkpoint:
        from aerojepa.sim.action_residual import load_residual_head

        residual_head = load_residual_head(
            args.residual_checkpoint, device, latent_dim=latent_dim
        )
        print(
            f"Loaded residual ({residual_head.num_params()} params) "
            f"from {args.residual_checkpoint}"
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stack_kw = dict(
        residual_head=residual_head,
        planner_mode=args.planner,
        latent_smooth=args.latent_smooth,
        latent_refine_steps=args.latent_refine_steps,
    )

    print(
        f"=== wind_gust @ {args.wind_mps} m/s "
        f"(planner={args.planner}, residual={'yes' if residual_head else 'no'}) ==="
    )
    wind = run_closed_loop_demo(
        model,
        device,
        task="wind_gust",
        out_dir=out_dir / "wind_gust",
        img_size=img_size,
        max_steps=args.wind_steps,
        seed=args.seed,
        wind_mps=args.wind_mps,
        wind_onset=args.wind_onset,
        **stack_kw,
    )

    print(f"\n=== aggressive_turn L-course {DEFAULT_AGGRESSIVE_LEG1} → {DEFAULT_AGGRESSIVE_LEG2} ===")
    turn = run_closed_loop_demo(
        model,
        device,
        task="aggressive_turn",
        out_dir=out_dir / "aggressive_turn",
        img_size=img_size,
        max_steps=args.turn_steps,
        seed=args.seed,
        action_scale=0.10,
        horizon=4,
        aggressive_leg1=DEFAULT_AGGRESSIVE_LEG1,
        aggressive_leg2=DEFAULT_AGGRESSIVE_LEG2,
        **stack_kw,
    )

    summary = {
        "world_checkpoint": args.checkpoint,
        "residual_checkpoint": args.residual_checkpoint,
        "planner_mode": args.planner,
        "latent_smooth": args.latent_smooth,
        "latent_refine_steps": args.latent_refine_steps,
        "seed": args.seed,
        "heuristic_action_map": "unchanged (fair comparison)",
        "wind_gust": {
            "wind_mps": args.wind_mps,
            "wind_onset": args.wind_onset,
            "max_steps": args.wind_steps,
            "policies": {n: _policy_row(ep) for n, ep in wind.results.items()},
            "metrics_path": str(wind.metrics_path),
            "plot_path": str(wind.plot_path),
            "gif_paths": {n: str(p) for n, p in wind.gif_paths.items()},
        },
        "aggressive_turn": {
            "leg1": list(DEFAULT_AGGRESSIVE_LEG1),
            "leg2": list(DEFAULT_AGGRESSIVE_LEG2),
            "max_steps": args.turn_steps,
            "policies": {n: _policy_row(ep) for n, ep in turn.results.items()},
            "metrics_path": str(turn.metrics_path),
            "plot_path": str(turn.plot_path),
            "gif_paths": {n: str(p) for n, p in turn.gif_paths.items()},
        },
        "breaking_points": _breaking_points(wind.results, turn.results, args.wind_mps),
    }
    summary_path = out_dir / "stress_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print("\n=== Breaking points ===")
    for line in summary["breaking_points"]["narrative"]:
        print(f"  • {line}")
    print(f"\nsummary → {summary_path}")


def _breaking_points(wind_results, turn_results, wind_mps: float) -> dict:
    """Human-readable failure summary from the two stress tasks."""
    narrative: list[str] = []
    table: dict[str, dict] = {"wind_gust": {}, "aggressive_turn": {}}

    for name, ep in wind_results.items():
        table["wind_gust"][name] = {
            "failure_mode": ep.failure_mode,
            "max_xy_drift": round(ep.max_xy_drift, 3),
            "survived": ep.survived,
        }
    for name, ep in turn_results.items():
        table["aggressive_turn"][name] = {
            "failure_mode": ep.failure_mode,
            "waypoints": f"{ep.waypoints_reached}/{ep.waypoints_total}",
            "survived": ep.survived,
        }

    wp = wind_results.get("planner")
    wh = wind_results.get("hover")
    if wp is not None:
        narrative.append(
            f"Wind {wind_mps} m/s - planner: {wp.failure_mode} "
            f"(max_xy={wp.max_xy_drift:.2f} m). {wp.failure_detail}"
        )
    if wh is not None:
        narrative.append(
            f"Wind {wind_mps} m/s - pure hover (no correction): {wh.failure_mode} "
            f"(max_xy={wh.max_xy_drift:.2f} m)."
        )
        if wp is not None and wp.max_xy_drift < wh.max_xy_drift:
            narrative.append(
                "Planner contains wind drift better than open-loop hover "
                f"({wp.max_xy_drift:.2f} < {wh.max_xy_drift:.2f} m)."
            )
        elif wp is not None:
            narrative.append(
                "Planner does NOT beat open-loop hover under wind - world-model "
                "station-keeping breaks here (heuristic map + latent plan insufficient)."
            )

    tp = turn_results.get("planner")
    ts = turn_results.get("seek")
    if tp is not None:
        narrative.append(
            f"Aggressive L-turn - planner: {tp.failure_mode} "
            f"(legs {tp.waypoints_reached}/{tp.waypoints_total}). {tp.failure_detail}"
        )
    if ts is not None:
        narrative.append(
            f"Aggressive L-turn - privileged seek PD: {ts.failure_mode} "
            f"(legs {ts.waypoints_reached}/{ts.waypoints_total})."
        )
        if tp is not None and (tp.waypoints_reached or 0) < (ts.waypoints_reached or 0):
            narrative.append(
                "Breaking point: sharp 90° corner - reactive seek clears more legs than "
                "the perception-driven planner (heuristic map + short horizon)."
            )
        elif tp is not None and tp.failure_mode == "ok" and ts.failure_mode == "ok":
            narrative.append("Both planner and seek clear the L-turn at this aggressiveness.")

    return {"table": table, "narrative": narrative}


if __name__ == "__main__":
    main()
