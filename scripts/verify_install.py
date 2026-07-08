#!/usr/bin/env python
"""Fast sanity check that the environment and code are wired up correctly.

Renders a synthetic clip, builds the model, runs one forward/backward pass, and
prints the parameter count. If this succeeds, everything else should too.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import torch

from aerojepa.data.synthetic import render_clip
from aerojepa.masking import build_mask_collator
from aerojepa.models.jepa import AeroJEPA
from aerojepa.train import stack_indices
from aerojepa.utils.config import load_config
from aerojepa.utils.device import get_device


def main() -> None:
    print("AeroJEPA install verification")
    print(f"  torch {torch.__version__}")
    device = get_device("auto")
    print(f"  device: {device}")

    cfg = load_config("configs/smoke_test.yaml")

    clip = render_clip(seed=0, num_frames=cfg["data"]["num_frames"], img_size=cfg["data"]["img_size"])
    print(f"  synthetic clip: frames={tuple(clip.frames.shape)}  actions={tuple(clip.actions.shape)}")

    model = AeroJEPA.from_config(cfg).to(device)
    print(f"  trainable params: {model.num_trainable_params():,}")

    clips = clip.frames.unsqueeze(0).to(device)
    grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = build_mask_collator(cfg, grid, model.encoder.num_temporal)
    masks = collator(1)
    ctx = stack_indices(masks.context_indices, device)
    tgt = stack_indices(masks.target_indices, device)

    out = model(clips, ctx, tgt)
    out["loss"].backward()
    print(f"  forward/backward OK  loss={out['loss'].item():.4f}")
    print("All checks passed.")


if __name__ == "__main__":
    main()
