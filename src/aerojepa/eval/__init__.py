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


def load_pretrained(
    name: str,
    device: torch.device,
    *,
    pretrained: bool = True,
) -> tuple[AeroJEPA, dict[str, Any]]:
    """Load a released registry checkpoint, downloading when URLs are configured.

    ``name`` must be one of ``world_model`` or ``real_finetune_fast``.
    If the local dest exists, it is loaded. Otherwise, when ``pretrained=True``
    and ``released_weights/urls.yaml`` is not a placeholder, the file is
    downloaded. Raises a short error if weights are not published yet.
    """
    from aerojepa.eval.weights import WeightDownloadError, ensure_checkpoint

    try:
        path = ensure_checkpoint(name, pretrained=pretrained)
    except WeightDownloadError as exc:
        raise RuntimeError(
            f"weights not published yet; train or pass --checkpoint ({exc})"
        ) from exc
    except FileNotFoundError as exc:
        raise RuntimeError(str(exc)) from exc
    return load_model(path, device)


__all__ = ["load_model", "load_pretrained"]
