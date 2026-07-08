from __future__ import annotations

from typing import Any

import torch

from aerojepa.eval import load_model
from aerojepa.eval.latent_pred import latent_prediction_metrics
from aerojepa.masking import build_mask_collator
from aerojepa.train import build_dataloaders_from_cfg


def compare_models(
    baseline_checkpoint: str,
    looped_checkpoint: str,
    device: torch.device,
    max_batches: int = 8,
) -> dict[str, Any]:
    """Score two checkpoints on the same held-out clips and report the delta."""
    results: dict[str, Any] = {}
    for name, ckpt in (("baseline", baseline_checkpoint), ("looped", looped_checkpoint)):
        model, cfg = load_model(ckpt, device)
        _, val_loader = build_dataloaders_from_cfg(cfg)
        grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
        collator = build_mask_collator(cfg, grid, model.encoder.num_temporal)
        results[name] = latent_prediction_metrics(model, val_loader, collator, device, cfg, max_batches)

    results["cosine_delta"] = results["looped"]["cosine"] - results["baseline"]["cosine"]
    return results
