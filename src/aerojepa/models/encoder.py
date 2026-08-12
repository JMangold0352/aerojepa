from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from aerojepa.models.vit import (
    Block,
    FramePatchEmbed,
    TubeletEmbed,
    build_pos_embed,
    init_pos_embed,
    spatiotemporal_pos_embed,
)


class VideoTransformerEncoder(nn.Module):
    """ViT encoder over a set of space-time tokens from a short video clip.

    A clip of shape ``(B, T, C, H, W)`` is tokenized into ``T' * S`` tokens
    (``T'`` temporal, ``S`` spatial per frame), each tagged with a factorized
    space-time position. The encoder can run over *all* tokens (the EMA teacher
    path) or over an arbitrary *subset* of token indices (the context path),
    which is exactly what JEPA-style masked prediction needs.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        num_frames: int = 8,
        embed_dim: int = 192,
        depth: int = 6,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        tokenizer: str = "frame",
        tubelet_size: int = 2,
        norm_factory: Callable[[int], nn.Module] | None = None,
    ) -> None:
        super().__init__()
        norm_factory = norm_factory or (lambda d: nn.LayerNorm(d))

        if tokenizer == "tubelet":
            self.patch_embed: nn.Module = TubeletEmbed(
                img_size, patch_size, in_chans, embed_dim, tubelet_size
            )
            self.num_temporal = num_frames // tubelet_size
        else:
            self.patch_embed = FramePatchEmbed(img_size, patch_size, in_chans, embed_dim)
            self.num_temporal = num_frames

        self.num_spatial = self.patch_embed.num_spatial
        self.num_tokens = self.num_temporal * self.num_spatial

        # Factorized space-time position tables: cheaper than one giant table and
        # easy to interpret (one axis is "where", the other is "when").
        self.spatial_pos = build_pos_embed(self.num_spatial, embed_dim)
        self.temporal_pos = build_pos_embed(self.num_temporal, embed_dim)
        init_pos_embed(self.spatial_pos)
        init_pos_embed(self.temporal_pos)

        dpr = [drop_path * i / max(1, depth - 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    norm_factory=norm_factory,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_factory(embed_dim)

    def _pos_embed(self) -> torch.Tensor:
        return spatiotemporal_pos_embed(self.spatial_pos, self.temporal_pos)

    def tokenize(self, clips: torch.Tensor) -> torch.Tensor:
        """Clips ``(B, T, C, H, W)`` -> flat tokens ``(B, num_tokens, D)`` + pos."""
        tokens = self.patch_embed(clips)  # (B, T', S, D)
        b = tokens.shape[0]
        tokens = tokens.reshape(b, self.num_tokens, -1)
        return tokens + self._pos_embed()

    def forward(self, clips: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        """Encode only ``token_indices`` (the visible context) of each clip."""
        tokens = self.tokenize(clips)
        idx = token_indices.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        gathered = torch.gather(tokens, 1, idx)
        for block in self.blocks:
            gathered = block(gathered)
        return self.norm(gathered)

    def forward_all_patches(self, clips: torch.Tensor) -> torch.Tensor:
        """Encode every space-time token; used by the EMA teacher and probes."""
        tokens = self.tokenize(clips)
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens)
