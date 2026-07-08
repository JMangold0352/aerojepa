"""Sim-to-real gap evaluation on a folder of real clips."""

from __future__ import annotations

import copy
from pathlib import Path

from aerojepa.eval import load_model
from aerojepa.eval.latent_pred import latent_prediction_metrics
from aerojepa.eval.rollout import rollout_metrics
from aerojepa.masking import build_mask_collator
from aerojepa.train import build_dataloaders_from_cfg
from aerojepa.utils.device import get_device


def _metrics_for(model, cfg, device, max_batches):
    _, val_loader = build_dataloaders_from_cfg(cfg)
    grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = build_mask_collator(cfg, grid, model.encoder.num_temporal)
    latent = latent_prediction_metrics(model, val_loader, collator, device, cfg, max_batches)
    roll = rollout_metrics(model, val_loader, device, cfg, max_batches=max_batches)
    return {"latent_prediction": latent, "rollout": roll}


def evaluate_real_gap(
    checkpoint: str | Path,
    data_dir: str | Path,
    *,
    device: str = "auto",
    max_batches: int = 8,
    window_mode: str = "uniform",
) -> dict:
    """Run synthetic-vs-real metrics for one checkpoint and return the report dict."""
    device_obj = get_device(device)
    model, cfg = load_model(str(checkpoint), device_obj)

    synth = _metrics_for(model, cfg, device_obj, max_batches)

    real_cfg = copy.deepcopy(cfg)
    real_cfg["data"]["source"] = "video"
    real_cfg["data"]["data_dir"] = str(data_dir)
    real_cfg["data"]["window_mode"] = window_mode
    real_cfg["data"].setdefault("num_workers", 0)
    real = _metrics_for(model, real_cfg, device_obj, max_batches)

    cos_gap = synth["latent_prediction"]["cosine"] - real["latent_prediction"]["cosine"]
    return {
        "checkpoint": str(checkpoint),
        "data_dir": str(data_dir),
        "synthetic": synth,
        "real": real,
        "gap": {
            "latent_cosine": cos_gap,
            "smooth_l1": real["latent_prediction"]["smooth_l1"]
            - synth["latent_prediction"]["smooth_l1"],
        },
    }
