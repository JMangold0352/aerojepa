from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aerojepa.models.jepa import AeroJEPA
from aerojepa.train import _prep_actions


def _build_future_mask(
    num_temporal: int, num_spatial: int, context_frames: int, target_frame: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    context = torch.arange(0, context_frames * num_spatial, device=device)
    start = target_frame * num_spatial
    target = torch.arange(start, start + num_spatial, device=device)
    return context, target


@torch.no_grad()
def rollout_metrics(
    model: AeroJEPA,
    loader: DataLoader,
    device: torch.device,
    cfg: dict[str, Any],
    context_frames: int | None = None,
    max_batches: int = 8,
) -> dict[str, list]:
    """How well the model predicts further into the future.

    Given the first ``context_frames`` of a clip, predict the latents of each
    subsequent frame and measure agreement with the teacher. Accuracy that
    degrades gracefully with horizon is the signature of a usable world model
    for planning and obstacle anticipation.
    """
    model.eval()
    num_temporal = model.encoder.num_temporal
    num_spatial = model.encoder.num_spatial
    if context_frames is None:
        context_frames = max(1, num_temporal // 2)
    use_actions = bool(cfg["predictor"].get("action_conditioning", False))

    horizons = list(range(1, num_temporal - context_frames + 1))
    cos_by_h = [0.0 for _ in horizons]
    l1_by_h = [0.0 for _ in horizons]
    n = 0

    for i, (clips, actions) in enumerate(loader):
        if i >= max_batches:
            break
        clips = clips.to(device)
        b = clips.shape[0]
        acts_full = _prep_actions(actions, num_temporal, device, cfg) if use_actions else None

        for hi, horizon in enumerate(horizons):
            target_frame = context_frames + horizon - 1
            ctx, tgt = _build_future_mask(num_temporal, num_spatial, context_frames, target_frame, device)
            ctx = ctx.unsqueeze(0).expand(b, -1)
            tgt = tgt.unsqueeze(0).expand(b, -1)
            out = model(clips, ctx, tgt, actions=acts_full)
            cos_by_h[hi] += float(F.cosine_similarity(out["pred_repr"], out["target_repr"], dim=-1).mean().item())
            l1_by_h[hi] += float(F.smooth_l1_loss(out["pred_repr"], out["target_repr"]).item())
        n += 1

    n = max(1, n)
    return {
        "horizon": horizons,
        "cosine": [c / n for c in cos_by_h],
        "smooth_l1": [l / n for l in l1_by_h],
        "context_frames": context_frames,
    }
