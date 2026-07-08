from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from aerojepa.models.jepa import AeroJEPA


def load_model(checkpoint_path: str | Path, device: torch.device) -> tuple[AeroJEPA, dict[str, Any]]:
    """Rebuild a model from a checkpoint and its embedded config.

    Checkpoints store the exact config they were trained with, so evaluation is
    always reproducible without having to remember which YAML was used.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg: dict[str, Any] = ckpt["config"]
    model = AeroJEPA.from_config(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


__all__ = ["load_model"]
