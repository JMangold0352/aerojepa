#!/usr/bin/env python
"""Hard PyFlyt success-vs-difficulty suite.

Keeps v1 stack only. Sweeps wind, L-turn scale, recover delay; ≥10 seeds.
Writes visualizations/closed_loop/stress_suite.json + success-vs-difficulty figure.

Example::

    python scripts/run_hard_pyflyt_suite.py \\
        --checkpoint checkpoints/action_conditioned_wilds/latest.pt \\
        --residual-checkpoint checkpoints/action_residual_wilds/best.pt \\
        --seeds 0-9 --quick
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

from aerojepa.eval import load_model
from aerojepa.sim.closed_loop import (
    DEFAULT_AGGRESSIVE_LEG1,
    DEFAULT_AGGRESSIVE_LEG2,
    run_closed_loop_episode,
)
from aerojepa.sim.action_residual import load_residual_head
from aerojepa.utils.device import get_device


def _parse_seeds(spec: str) -> list[int]:
    """Parse '0-9' or '0,1,2' into a list of ints."""
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/action_conditioned_wilds/latest.pt",
    )
    parser.add_argument(
        "--residual-checkpoint",
        default="checkpoints/action_residual_wilds/best.pt",
    )
    parser.add_argument("--planner", default="gradient", choices=["shooting", "gradient"])
    parser.add_argument("--latent-smooth", type=float, default=0.05)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fewer difficulty levels / shorter episodes for a smoke run.",
    )
    parser.add_argument(
        "--out",
        default="visualizations/closed_loop/stress_suite.json",
    )
    parser.add_argument(
        "--figure",
        default="visualizations/closed_loop/stress_suite_success.png",
    )
    parser.add_argument("--policy", default="planner")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    if len(seeds) < 10 and not args.quick:
        print(f"warning: {len(seeds)} seeds < 10; prefer ≥10 for CIs")

    device = get_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    img_size = int(cfg["data"]["img_size"])
    latent_dim = int(cfg["encoder"]["embed_dim"])
    residual = load_residual_head(args.residual_checkpoint, device, latent_dim=latent_dim)

    wind_levels = [0.0, 1.0, 2.0, 3.0, 4.0] if not args.quick else [0.0, 2.0, 4.0]
    turn_scales = [0.5, 0.75, 1.0, 1.25] if not args.quick else [0.5, 1.0]
    recover_delays = [10, 20, 40, 60] if not args.quick else [20, 40]
    hover_disturbs = [0.5, 0.75, 1.0, 1.25] if not args.quick else [0.75, 1.0]

    stack = dict(
        residual_head=residual,
        planner_mode=args.planner,
        latent_smooth=args.latent_smooth,
        latent_refine_steps=8,
    )

    curves: dict[str, list[dict]] = {
        "wind_gust": [],
        "aggressive_turn": [],
        "recover_delay": [],
        "hover_disturb": [],
    }
    failures: list[dict] = []

    # --- Wind sweep ---
    for w in wind_levels:
        oks = []
        for seed in seeds:
            ep = run_closed_loop_episode(
                model,
                device,
                task="wind_gust",
                policy=args.policy,
                img_size=img_size,
                max_steps=120 if args.quick else 200,
                seed=seed,
                wind_mps=w,
                wind_onset=30,
                record_frames=False,
                **stack,
            )
            ok = bool(ep.survived) and ep.failure_mode == "ok"
            oks.append(ok)
            if not ok:
                failures.append(
                    {
                        "task": "wind_gust",
                        "difficulty": w,
                        "seed": seed,
                        "failure_mode": ep.failure_mode,
                        "max_xy_drift": ep.max_xy_drift,
                    }
                )
        rate = float(np.mean(oks))
        curves["wind_gust"].append(
            {
                "difficulty": w,
                "difficulty_label": f"{w:.1f} m/s",
                "success_rate": rate,
                "n_seeds": len(seeds),
                "se": float(np.sqrt(rate * (1 - rate) / max(len(seeds), 1))),
            }
        )
        print(f"wind {w:.1f} m/s: success={rate:.0%} ({sum(oks)}/{len(seeds)})")

    # --- L-turn scale ---
    for scale in turn_scales:
        leg1 = tuple(float(x) * scale for x in DEFAULT_AGGRESSIVE_LEG1)
        leg2 = tuple(float(x) * scale for x in DEFAULT_AGGRESSIVE_LEG2)
        oks = []
        for seed in seeds:
            ep = run_closed_loop_episode(
                model,
                device,
                task="aggressive_turn",
                policy=args.policy,
                img_size=img_size,
                max_steps=160 if args.quick else 260,
                seed=seed,
                action_scale=0.10,
                aggressive_leg1=leg1,
                aggressive_leg2=leg2,
                record_frames=False,
                **stack,
            )
            ok = bool(ep.survived) and ep.failure_mode == "ok" and bool(ep.reached)
            oks.append(ok)
            if not ok:
                failures.append(
                    {
                        "task": "aggressive_turn",
                        "difficulty": scale,
                        "seed": seed,
                        "failure_mode": ep.failure_mode,
                        "reached": ep.reached,
                    }
                )
        rate = float(np.mean(oks))
        curves["aggressive_turn"].append(
            {
                "difficulty": scale,
                "difficulty_label": f"scale×{scale:.2f}",
                "success_rate": rate,
                "n_seeds": len(seeds),
                "se": float(np.sqrt(rate * (1 - rate) / max(len(seeds), 1))),
            }
        )
        print(f"L-turn ×{scale:.2f}: success={rate:.0%} ({sum(oks)}/{len(seeds)})")

    # --- Recover delay ---
    for delay in recover_delays:
        oks = []
        for seed in seeds:
            ep = run_closed_loop_episode(
                model,
                device,
                task="recover",
                policy=args.policy,
                img_size=img_size,
                max_steps=160 if args.quick else 220,
                seed=seed,
                disturb_at=delay,
                record_frames=False,
                **stack,
            )
            ok = bool(ep.survived) and ep.failure_mode == "ok"
            oks.append(ok)
            if not ok:
                failures.append(
                    {
                        "task": "recover",
                        "difficulty": delay,
                        "seed": seed,
                        "failure_mode": ep.failure_mode,
                    }
                )
        rate = float(np.mean(oks))
        curves["recover_delay"].append(
            {
                "difficulty": delay,
                "difficulty_label": f"delay={delay}",
                "success_rate": rate,
                "n_seeds": len(seeds),
                "se": float(np.sqrt(rate * (1 - rate) / max(len(seeds), 1))),
            }
        )
        print(f"recover delay={delay}: success={rate:.0%} ({sum(oks)}/{len(seeds)})")

    # --- Hover disturb magnitude (kick strength via action scale on recover kick) ---
    for mag in hover_disturbs:
        kick = (0.0, float(0.75 * mag), 0.0, 0.39)
        oks = []
        for seed in seeds:
            ep = run_closed_loop_episode(
                model,
                device,
                task="recover",
                policy=args.policy,
                img_size=img_size,
                max_steps=160 if args.quick else 220,
                seed=seed,
                disturb_at=30,
                disturb_action=kick,
                record_frames=False,
                **stack,
            )
            ok = bool(ep.survived) and ep.failure_mode == "ok"
            oks.append(ok)
            if not ok:
                failures.append(
                    {
                        "task": "hover_disturb",
                        "difficulty": mag,
                        "seed": seed,
                        "failure_mode": ep.failure_mode,
                    }
                )
        rate = float(np.mean(oks))
        curves["hover_disturb"].append(
            {
                "difficulty": mag,
                "difficulty_label": f"kick×{mag:.2f}",
                "success_rate": rate,
                "n_seeds": len(seeds),
                "se": float(np.sqrt(rate * (1 - rate) / max(len(seeds), 1))),
            }
        )
        print(f"hover disturb ×{mag:.2f}: success={rate:.0%} ({sum(oks)}/{len(seeds)})")

    summary = {
        "world_checkpoint": args.checkpoint,
        "residual_checkpoint": args.residual_checkpoint,
        "planner": args.planner,
        "latent_smooth": args.latent_smooth,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "curves": curves,
        "n_failures_logged": len(failures),
        "failures_sample": failures[:40],
        "note": "Success vs difficulty, 10 seeds, v1 stack only.",
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    mapping = [
        ("wind_gust", "Wind (m/s)", axes[0, 0]),
        ("aggressive_turn", "L-turn scale", axes[0, 1]),
        ("recover_delay", "Recover disturb_at", axes[1, 0]),
        ("hover_disturb", "Hover kick scale", axes[1, 1]),
    ]
    for key, xlabel, ax in mapping:
        pts = curves[key]
        xs = [p["difficulty"] for p in pts]
        ys = [p["success_rate"] for p in pts]
        se = [p["se"] for p in pts]
        ax.errorbar(xs, ys, yerr=se, marker="o", capsize=3)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("success rate")
        ax.set_title(key)
        ax.axhline(1.0, color="gray", lw=0.5, ls="--")
    fig.suptitle(
        f"Hard PyFlyt - v1 stack, {len(seeds)} seeds",
        fontsize=11,
    )
    fig.tight_layout()
    fig_path = Path(args.figure)
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
