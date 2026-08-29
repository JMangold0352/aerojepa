#!/usr/bin/env python
"""Four-way L-turn action ablation (true / zero / shuffle / residual-off).

Asks whether the world-model predictor is doing anything on aggressive_turn
at scale 1.0 and the published ×1.25 cliff. Does not retune seek blends or
retrain. v1 stack only.

Predictor ablations match ``scripts/eval_action_counterfactual.py``: zero or
batch-shuffle the action tensor the *predictor* sees. Planned 6-DoF actions
still map to PyFlyt via ``aerojepa_to_pyflyt`` (+ residual when enabled).

Example::

    python scripts/eval_lturn_action_ablation.py \\
        --checkpoint checkpoints/action_conditioned_wilds/latest.pt \\
        --residual-checkpoint checkpoints/action_residual_wilds/best.pt \\
        --seeds 0-9
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

from aerojepa.eval import load_model
from aerojepa.sim.action_residual import load_residual_head
from aerojepa.sim.closed_loop import (
    DEFAULT_AGGRESSIVE_LEG1,
    DEFAULT_AGGRESSIVE_LEG2,
    run_closed_loop_episode,
)
from aerojepa.utils.device import get_device


def _parse_seeds(spec: str) -> list[int]:
    spec = spec.strip()
    if "-" in spec and "," not in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip()]


def _verdict(by_condition: dict[str, dict]) -> str:
    """Compare success at the hard cliff (scale 1.25) across true/zero/shuffle."""
    cliff = {}
    for cond in ("true", "zero", "shuffle"):
        key = f"{cond}@1.25"
        row = by_condition.get(key) or by_condition.get(cond, {}).get("1.25")
        if row is None:
            # nested layout
            continue
        cliff[cond] = float(row["success_rate"])

    # Support nested {cond: {scale: ...}}
    if len(cliff) < 3:
        cliff = {}
        for cond in ("true", "zero", "shuffle"):
            block = by_condition.get(cond, {})
            if "1.25" in block:
                cliff[cond] = float(block["1.25"]["success_rate"])

    if len(cliff) < 3:
        return "incomplete: missing true/zero/shuffle at scale 1.25"

    t, z, s = cliff["true"], cliff["zero"], cliff["shuffle"]
    spread = max(t, z, s) - min(t, z, s)
    if spread <= 0.15 and abs(t - z) <= 0.15 and abs(t - s) <= 0.15:
        return (
            "true ≈ zero ≈ shuffle on the ×1.25 L-turn: the world model is not "
            "the lever. Hardware next step is shadow/observer, not commanded flight."
        )
    if t > s + 0.2 and t > z + 0.1:
        return (
            "true clearly beats shuffle on the ×1.25 L-turn: F2 is starting to "
            "flip. Still not a causal controller claim."
        )
    return (
        f"mixed at ×1.25 (true={t:.0%}, zero={z:.0%}, shuffle={s:.0%}): "
        "no clean causal win; do not claim a causal controller."
    )


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
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument(
        "--out",
        default="results/lturn_action_ablation.json",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smoke: 2 seeds only.",
    )
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    if args.quick:
        seeds = seeds[:2]
    if len(seeds) < 10 and not args.quick:
        print(f"warning: {len(seeds)} seeds < 10; prefer 0-9 to match stress suite")

    device = get_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    img_size = int(cfg["data"]["img_size"])
    latent_dim = int(cfg["encoder"]["embed_dim"])
    residual = load_residual_head(args.residual_checkpoint, device, latent_dim=latent_dim)

    conditions = (
        ("true", "true", residual),
        ("zero", "zero", residual),
        ("shuffle", "shuffle", residual),
        ("residual_off", "true", None),
    )
    scales = (1.0, 1.25)

    protocol = {
        "predictor_action_tensor": (
            "LatentPlanner feeds actions_full of shape (N_candidates, num_temporal, 6) "
            "into the action-conditioned predictor. Ablations: true=pass-through; "
            "zero=zeros_like(actions_full); shuffle=permute across candidate batch "
            "(perm fixed for one plan() call), same batch-shuffle idea as "
            "eval_action_counterfactual.py. Executed PyFlyt controls still come from "
            "planned best_actions via aerojepa_to_pyflyt (+ residual when enabled)."
        ),
        "residual_off": "residual_head=None; heuristic aerojepa_to_pyflyt only.",
        "stack": "v1 action_conditioned_wilds + action_residual_wilds (except residual_off).",
        "task": "aggressive_turn",
        "scales": list(scales),
        "legs_at_scale_1": {
            "leg1": list(DEFAULT_AGGRESSIVE_LEG1),
            "leg2": list(DEFAULT_AGGRESSIVE_LEG2),
        },
        "seeds": seeds,
        "planner": args.planner,
        "latent_smooth": args.latent_smooth,
        "no_seek_retune": True,
    }

    by_condition: dict[str, dict] = {}
    episodes: list[dict] = []

    for cond_name, ablation, res_head in conditions:
        by_condition[cond_name] = {}
        for scale in scales:
            leg1 = tuple(float(x) * scale for x in DEFAULT_AGGRESSIVE_LEG1)
            leg2 = tuple(float(x) * scale for x in DEFAULT_AGGRESSIVE_LEG2)
            oks: list[bool] = []
            modes: Counter[str] = Counter()
            loop_ms: list[float] = []
            holds = 0
            for seed in seeds:
                print(f"{cond_name} scale={scale} seed={seed} ...", flush=True)
                ep = run_closed_loop_episode(
                    model,
                    device,
                    policy="planner",
                    task="aggressive_turn",
                    img_size=img_size,
                    max_steps=args.max_steps,
                    seed=seed,
                    record_frames=False,
                    residual_head=res_head,
                    planner_mode=args.planner,
                    latent_smooth=args.latent_smooth,
                    latent_refine_steps=8,
                    aggressive_leg1=leg1,
                    aggressive_leg2=leg2,
                    predictor_action_ablation=ablation,
                )
                ok = ep.failure_mode == "ok"
                oks.append(ok)
                modes[str(ep.failure_mode or "unknown")] += 1
                if ep.mean_loop_ms is not None:
                    loop_ms.append(float(ep.mean_loop_ms))
                holds += int(ep.watchdog_holds)
                episodes.append(
                    {
                        "condition": cond_name,
                        "scale": scale,
                        "seed": seed,
                        "success": ok,
                        "failure_mode": ep.failure_mode,
                        "failure_detail": ep.failure_detail,
                        "waypoints_reached": ep.waypoints_reached,
                        "waypoints_total": ep.waypoints_total,
                        "mean_loop_ms": ep.mean_loop_ms,
                        "watchdog_holds": ep.watchdog_holds,
                        "survived": ep.survived,
                    }
                )
            n = max(1, len(oks))
            by_condition[cond_name][f"{scale:g}"] = {
                "success_rate": float(sum(oks) / n),
                "n": len(oks),
                "failure_mode_hist": dict(modes),
                "mean_loop_ms": float(sum(loop_ms) / len(loop_ms)) if loop_ms else None,
                "watchdog_holds_total": holds,
                "leg1": list(leg1),
                "leg2": list(leg2),
            }

    verdict = _verdict(by_condition)
    report = {
        "protocol": protocol,
        "by_condition": by_condition,
        "verdict": verdict,
        "episodes": episodes,
        "checkpoint": args.checkpoint,
        "residual_checkpoint": args.residual_checkpoint,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out}")
    print(f"Verdict: {verdict}")
    for cond, scales_d in by_condition.items():
        for sc, row in scales_d.items():
            print(
                f"  {cond:12s} ×{sc}: success={row['success_rate']:.0%} "
                f"modes={row['failure_mode_hist']}"
            )


if __name__ == "__main__":
    main()
