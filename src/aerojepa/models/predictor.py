from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from aerojepa.models.vit import (
    Block,
    BlockStack,
    build_pos_embed,
    init_pos_embed,
    rms_norm_factory,
    spatiotemporal_pos_embed,
)


class VideoPredictor(nn.Module):
    """Narrow space-time ViT predictor, built as a reusable ``BlockStack``.

    Given the encoded *context* tokens, the predictor places learned mask tokens
    at the *target* space-time positions and predicts the target latents that the
    EMA teacher would produce there. Structuring the transformer as a single
    ``BlockStack`` is what lets the looped predictor re-apply the same weights
    for several refinement steps at zero extra parameter cost.

    Optional 6-DoF action conditioning adds a per-frame motion embedding to every
    token, turning the predictor from "fill in the blanks" into a genuine world
    model: *given where the drone moved, what will the scene look like next?*
    """

    def __init__(
        self,
        num_temporal: int = 8,
        num_spatial: int = 64,
        encoder_dim: int = 192,
        predictor_dim: int = 96,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_factory: Callable[[int], nn.Module] | None = None,
        sandwich_norm: bool = False,
        ffn_type: str = "mlp",
        action_dim: int = 0,
        action_conditioning: bool = False,
    ) -> None:
        super().__init__()
        norm_factory = norm_factory or (lambda d: nn.LayerNorm(d))
        self.num_temporal = num_temporal
        self.num_spatial = num_spatial
        self.num_tokens = num_temporal * num_spatial
        self.predictor_dim = predictor_dim
        self.action_conditioning = action_conditioning and action_dim > 0

        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.spatial_pos = build_pos_embed(num_spatial, predictor_dim)
        self.temporal_pos = build_pos_embed(num_temporal, predictor_dim)
        init_pos_embed(self.spatial_pos)
        init_pos_embed(self.temporal_pos)

        self.ctx_proj = nn.Linear(encoder_dim, predictor_dim)
        self.action_proj = (
            nn.Linear(action_dim, predictor_dim) if self.action_conditioning else None
        )

        blocks = nn.ModuleList(
            [
                Block(
                    predictor_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    norm_factory=norm_factory,
                    sandwich_norm=sandwich_norm,
                    ffn_type=ffn_type,
                )
                for _ in range(depth)
            ]
        )
        self.block_stack = BlockStack(blocks)
        self.norm = norm_factory(predictor_dim)
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

    def _pos_embed(self) -> torch.Tensor:
        return spatiotemporal_pos_embed(self.spatial_pos, self.temporal_pos)

    def _action_embed(self, actions: torch.Tensor) -> torch.Tensor:
        """Expand per-frame actions ``(B, T, action_dim)`` to per-token ``(B, N, D)``."""
        b = actions.shape[0]
        assert self.action_proj is not None
        frame_emb = self.action_proj(actions)  # (B, T, D)
        frame_emb = frame_emb.unsqueeze(2).expand(b, self.num_temporal, self.num_spatial, -1)
        return frame_emb.reshape(b, self.num_tokens, self.predictor_dim)

    def build_sequence(
        self,
        context_repr: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Assemble predictor input: context tokens + mask tokens at targets."""
        b = context_repr.shape[0]
        pos = self._pos_embed().expand(b, -1, -1)
        n_ctx = context_indices.shape[1]
        n_tgt = target_indices.shape[1]

        ctx_pos = torch.gather(
            pos, 1, context_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        )
        tgt_pos = torch.gather(
            pos, 1, target_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
        )

        x = context_repr.new_zeros(b, n_ctx + n_tgt, self.predictor_dim)
        x[:, :n_ctx] = self.ctx_proj(context_repr) + ctx_pos
        x[:, n_ctx:] = self.mask_token.expand(b, n_tgt, -1) + tgt_pos

        if self.action_conditioning and actions is not None:
            action_emb = self._action_embed(actions)
            ctx_act = torch.gather(
                action_emb, 1, context_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
            )
            tgt_act = torch.gather(
                action_emb, 1, target_indices.unsqueeze(-1).expand(-1, -1, self.predictor_dim)
            )
            x[:, :n_ctx] = x[:, :n_ctx] + ctx_act
            x[:, n_ctx:] = x[:, n_ctx:] + tgt_act
        return x

    def forward_stack(self, x: torch.Tensor) -> torch.Tensor:
        """One pass through the shared block stack (re-used by the looped predictor)."""
        x = self.block_stack(x)
        return self.norm(x)

    def forward(
        self,
        context_repr: torch.Tensor,
        context_indices: torch.Tensor,
        target_indices: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n_ctx = context_indices.shape[1]
        x = self.build_sequence(context_repr, context_indices, target_indices, actions)
        x = self.forward_stack(x)
        return self.output_proj(x[:, n_ctx:])

    @classmethod
    def world_model(
        cls,
        num_temporal: int = 8,
        num_spatial: int = 64,
        encoder_dim: int = 192,
        predictor_dim: int = 96,
        depth: int = 4,
        num_heads: int = 3,
        action_dim: int = 0,
        action_conditioning: bool = False,
    ) -> VideoPredictor:
        """RMSNorm + SwiGLU + sandwich norm recipe used by the looped world model."""
        return cls(
            num_temporal=num_temporal,
            num_spatial=num_spatial,
            encoder_dim=encoder_dim,
            predictor_dim=predictor_dim,
            depth=depth,
            num_heads=num_heads,
            norm_factory=rms_norm_factory,
            sandwich_norm=True,
            ffn_type="swiglu",
            action_dim=action_dim,
            action_conditioning=action_conditioning,
        )
