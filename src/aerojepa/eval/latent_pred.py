from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aerojepa.models.jepa import AeroJEPA
from aerojepa.train import _prep_actions, stack_indices


@torch.no_grad()
def latent_prediction_metrics(
    model: AeroJEPA,
    loader: DataLoader,
    collator,
    device: torch.device,
    cfg: dict[str, Any],
    max_batches: int = 8,
) -> dict[str, float]:
    """Quality of the predictor's guess about hidden/future latents.

    Returns mean cosine similarity (higher is better; 1.0 = perfect direction
    match) and mean smooth-L1 distance (lower is better) between predicted and
    EMA-teacher target latents on held-out clips.
    """
    model.eval()
    num_temporal = model.encoder.num_temporal
    use_actions = bool(cfg["predictor"].get("action_conditioning", False))

    cos_sum, l1_sum, n = 0.0, 0.0, 0
    for i, (clips, actions) in enumerate(loader):
        if i >= max_batches:
            break
        clips = clips.to(device)
        masks = collator(clips.shape[0])
        ctx = stack_indices(masks.context_indices, device)
        tgt = stack_indices(masks.target_indices, device)
        acts = _prep_actions(actions, num_temporal, device, cfg) if use_actions else None

        out = model(clips, ctx, tgt, actions=acts)
        cos = F.cosine_similarity(out["pred_repr"], out["target_repr"], dim=-1).mean()
        l1 = F.smooth_l1_loss(out["pred_repr"], out["target_repr"])
        cos_sum += float(cos.item())
        l1_sum += float(l1.item())
        n += 1

    n = max(1, n)
    return {"cosine": cos_sum / n, "smooth_l1": l1_sum / n}
