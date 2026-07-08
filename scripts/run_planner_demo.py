#!/usr/bin/env python
"""Run the AeroJEPA latent-space planner and render the result.

Imagines many action plans with a trained world model, picks the lowest-cost
one, executes it in the synthetic camera simulator, and saves an annotated GIF
(observed context -> planned rollout) plus a candidate-trajectory figure.

Examples::

    # Hover task with the action-conditioned checkpoint (true action selection):
    python scripts/run_planner_demo.py \
        --checkpoint checkpoints/action_conditioned/latest.pt --task hover

    # Waypoint task with the world-model checkpoint:
    python scripts/run_planner_demo.py \
        --checkpoint checkpoints/world_model/latest.pt --task waypoint \
        --goal 0.3 0.0 0.0

Without --checkpoint it runs an UNTRAINED smoke model so the pipeline still works.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import torch

from aerojepa.eval import load_model
from aerojepa.models.jepa import AeroJEPA
from aerojepa.sim.rollout_demo import plan_and_render
from aerojepa.utils.config import load_config
from aerojepa.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Trained checkpoint (optional).")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--task", default="hover", choices=["hover", "waypoint", "smoothness"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--num-candidates", type=int, default=64)
    parser.add_argument("--action-scale", type=float, default=0.06)
    parser.add_argument("--goal", type=float, nargs=3, default=None, metavar=("DX", "DY", "DZ"),
                        help="Waypoint goal displacement (waypoint task).")
    parser.add_argument("--out-dir", default="visualizations/planner")
    args = parser.parse_args()

    device = get_device(args.device)
    if args.checkpoint and Path(args.checkpoint).exists():
        model, cfg = load_model(args.checkpoint, device)
        print(f"Loaded {args.checkpoint}")
    else:
        cfg = load_config("configs/smoke_test.yaml")
        model = AeroJEPA.from_config(cfg).to(device).eval()
        print("No checkpoint given -> UNTRAINED smoke model (pipeline demo only).")

    conditioned = model.predictor_is_action_conditioned()
    print(f"Action-conditioned: {conditioned}  |  task: {args.task}")
    if not conditioned:
        print("  (plain world model: candidates scored by kinematic cost, not the latents.)")

    out_dir = Path(args.out_dir)
    out = plan_and_render(
        model,
        device,
        img_size=cfg["data"]["img_size"],
        in_chans=cfg["data"].get("in_chans", 3),
        num_obstacles=cfg["data"].get("num_obstacles", 5),
        seed=args.seed,
        task=args.task,
        context_frames=args.context_frames,
        horizon=args.horizon,
        num_candidates=args.num_candidates,
        action_scale=args.action_scale,
        goal=tuple(args.goal) if args.goal is not None else None,
        out_gif=out_dir / f"plan_{args.task}.gif",
        out_fig=out_dir / f"plan_{args.task}_trajectories.png",
    )

    r = out.result
    print(
        f"\nPlanned {r.horizon} steps from {r.context_frames} context frames "
        f"over {args.num_candidates} candidates."
    )
    print(f"  best cost      : {float(r.costs[r.best_index]):.4f}")
    print(f"  imagined coherence: {r.coherence:.3f}")
    print(f"  GIF  -> {out.gif_path}")
    print(f"  plot -> {out_dir / f'plan_{args.task}_trajectories.png'}")


if __name__ == "__main__":
    main()
