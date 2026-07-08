from __future__ import annotations

from typing import Any

import torch

from aerojepa.models.jepa import AeroJEPA
from aerojepa.models.looped_predictor import LoopedVideoPredictor


@torch.no_grad()
def per_frame_latents(model: AeroJEPA, clip: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Mean latent per frame for a single clip ``(T, C, H, W)`` -> ``(T, D)``.

    We encode the full clip with the (student) encoder and average each frame's
    spatial tokens, giving one vector per frame -- the input to the latent
    trajectory plot.
    """
    model.eval()
    tokens = model.encoder.forward_all_patches(clip.unsqueeze(0).to(device))  # (1, N, D)
    num_temporal = model.encoder.num_temporal
    num_spatial = model.encoder.num_spatial
    tokens = tokens.reshape(1, num_temporal, num_spatial, -1)
    return tokens.mean(dim=2)[0].cpu()


@torch.no_grad()
def collect_predictor_attention(
    model: AeroJEPA,
    clip: torch.Tensor,
    context_indices: torch.Tensor,
    target_indices: torch.Tensor,
    device: torch.device,
    actions: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the predictor with attention capture and return the last block's map.

    Returns a ``(N, N)`` attention matrix averaged over heads (and the single
    batch element), where ``N = len(context) + len(target)``.
    """
    model.eval()
    predictor = model.predictor
    base = predictor.base_predictor if isinstance(predictor, LoopedVideoPredictor) else predictor

    for block in base.block_stack.blocks:
        block.attn.store_attn = True

    ctx = context_indices.unsqueeze(0).to(device)
    tgt = target_indices.unsqueeze(0).to(device)
    context_repr = model.encoder(clip.unsqueeze(0).to(device), ctx)
    acts = actions.unsqueeze(0).to(device) if actions is not None else None
    predictor(context_repr, ctx, tgt, acts)

    attn = base.block_stack.blocks[-1].attn.last_attn  # (1, heads, N, N)
    for block in base.block_stack.blocks:
        block.attn.store_attn = False
        block.attn.last_attn = None
    return attn[0].mean(dim=0).cpu()


def variant_scores_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    """Pull a ``{variant: cosine}`` map out of an ablation summary JSON."""
    return {name: entry["cosine"] for name, entry in summary.items() if "cosine" in entry}
