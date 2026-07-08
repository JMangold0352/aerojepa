#!/usr/bin/env python
"""Regenerate the full AeroJEPA figure suite from a trained checkpoint.

Run via the wrapper (``python scripts/visualize.py --checkpoint ...``) or
directly. ``--fast`` uses fewer clips for a quick smoke render.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from aerojepa.data.synthetic import render_clip
from aerojepa.eval import load_model
from aerojepa.eval.loop_metrics import loop_metrics
from aerojepa.eval.rollout import rollout_metrics
from aerojepa.masking import build_mask_collator
from aerojepa.models.looped_predictor import LoopedVideoPredictor
from aerojepa.train import build_dataloaders_from_cfg
from aerojepa.utils.device import get_device
from aerojepa.viz import plots
from visualizations.inference import collect_predictor_attention, per_frame_latents


def generate_all(checkpoint: str, out_dir: str, device: str = "auto", fast: bool = False) -> None:
    dev = get_device(device)
    model, cfg = load_model(checkpoint, dev)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    max_batches = 2 if fast else 8

    _, val_loader = build_dataloaders_from_cfg(cfg)
    grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = build_mask_collator(cfg, grid, model.encoder.num_temporal)

    # 1. What the model sees.
    clip = render_clip(
        seed=12345, num_frames=cfg["data"]["num_frames"], img_size=cfg["data"]["img_size"]
    )
    plots.plot_clip_frames(clip.frames, out / "00_clip.png", "Example drone clip (held out)")

    # 2. World-model rollout accuracy vs horizon.
    roll = rollout_metrics(model, val_loader, dev, cfg, max_batches=max_batches)
    plots.plot_rollout_curve(roll, out / "01_rollout.png")

    # 3-4. Recurrence-specific figures.
    if isinstance(model.predictor, LoopedVideoPredictor):
        loops = loop_metrics(model, val_loader, collator, dev, cfg, max_batches=max_batches)
        plots.plot_per_loop_cosine(loops, out / "02_per_loop_cosine.png")
        plots.plot_exit_distribution(loops, out / "03_exit_distribution.png")

    # 5. Latent trajectory over time for one clip.
    traj = per_frame_latents(model, clip.frames, dev)
    plots.plot_latent_trajectory(traj, out / "04_latent_trajectory.png")

    # 6. Predictor attention, with frame boundaries.
    masks = collator(1)
    attn = collect_predictor_attention(
        model, clip.frames, masks.context_indices[0], masks.target_indices[0], dev
    )
    plots.plot_attention_matrix(
        attn, model.encoder.num_temporal, len(masks.context_indices[0]), out / "05_attention.png"
    )

    print(f"Wrote figures to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out-dir", default="visualizations/figures")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()
    generate_all(args.checkpoint, args.out_dir, device=args.device, fast=args.fast)


if __name__ == "__main__":
    main()
