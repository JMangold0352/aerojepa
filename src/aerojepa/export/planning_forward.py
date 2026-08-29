"""Frozen encoder+predictor forward used by LatentPlanner (exportable).

Matches the closed-loop imagine path: pad context clip to T frames, encode
context tokens, run the predictor with a full ``(B, T, action_dim)`` action
tensor. Residual heads and PyFlyt are intentionally excluded.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from aerojepa.models.jepa import AeroJEPA
from aerojepa.models.looped_predictor import LoopedVideoPredictor


class PlanningForward(nn.Module):
    """Scriptable / ONNX-friendly planning forward.

    Inputs
    ------
    context_clip : ``(B, context_frames, 3, H, W)`` float32 in ``[0, 1]``
    actions_full : ``(B, num_temporal, action_dim)`` float32
        Full clip-aligned actions (context + horizon). Zeros are fine for
        non-action-conditioned predictors.

    Output
    ------
    pred : ``(B, horizon, num_spatial, encoder_dim)``
    """

    def __init__(self, model: AeroJEPA, context_frames: int) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.predictor = model.predictor
        self.context_frames = int(context_frames)
        self.num_temporal = int(model.encoder.num_temporal)
        self.num_spatial = int(model.encoder.num_spatial)
        base_pred = (
            model.predictor.base_predictor
            if isinstance(model.predictor, LoopedVideoPredictor)
            else model.predictor
        )
        self.encoder_dim = int(base_pred.output_proj.out_features)
        self.horizon = self.num_temporal - self.context_frames
        if self.horizon <= 0:
            raise ValueError(
                f"context_frames={context_frames} must be < num_temporal={self.num_temporal}"
            )
        self.looped = isinstance(model.predictor, LoopedVideoPredictor)
        self.action_conditioned = bool(model.predictor_is_action_conditioned())
        # Freeze for export / bench.
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def _context_target_indices(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        n_ctx = self.context_frames * self.num_spatial
        n_all = self.num_temporal * self.num_spatial
        ctx = torch.arange(0, n_ctx, device=device).unsqueeze(0).expand(batch, -1)
        tgt = torch.arange(n_ctx, n_all, device=device).unsqueeze(0).expand(batch, -1)
        return ctx, tgt

    def forward(self, context_clip: torch.Tensor, actions_full: torch.Tensor) -> torch.Tensor:
        # context_clip: (B, C, 3, H, W)
        b = context_clip.shape[0]
        device = context_clip.device
        pad = context_clip[:, -1:].expand(b, self.horizon, -1, -1, -1)
        full_clip = torch.cat([context_clip, pad], dim=1)
        ctx, tgt = self._context_target_indices(b, device)
        context_repr = self.encoder(full_clip, ctx)
        acts = actions_full if self.action_conditioned else None
        out = self.predictor(context_repr, ctx, tgt, acts)
        if self.looped:
            out = out[0]
        return out.reshape(b, self.horizon, self.num_spatial, self.encoder_dim)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_planning_forward(
    model: AeroJEPA,
    *,
    context_frames: int | None = None,
) -> PlanningForward:
    """Build a planning forward; default context = closed-loop ``T // 2``."""
    t = int(model.encoder.num_temporal)
    if context_frames is None:
        context_frames = max(1, t // 2)
    return PlanningForward(model, context_frames=context_frames)


def planning_shapes(
    model: AeroJEPA,
    *,
    context_frames: int | None = None,
    img_size: int = 64,
) -> dict[str, Any]:
    """Shapes matching closed-loop defaults."""
    pf = build_planning_forward(model, context_frames=context_frames)
    img = int(img_size)
    action_dim = 6
    return {
        "img_size": img,
        "context_frames": pf.context_frames,
        "horizon": pf.horizon,
        "num_temporal": pf.num_temporal,
        "num_spatial": pf.num_spatial,
        "encoder_dim": pf.encoder_dim,
        "action_dim": action_dim,
        "action_conditioned": pf.action_conditioned,
        "looped": pf.looped,
        "param_count": pf.param_count(),
    }
