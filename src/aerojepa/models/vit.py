from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# These building blocks are inherited, largely unchanged, from the parent
# Looped-JEPA image model. Keeping them identical is deliberate: it lets us make
# an honest apples-to-apples claim that any gain in AeroJEPA comes from the
# *temporal* extensions, not from a different transformer implementation.


def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Stochastic depth: randomly zero whole residual branches per sample."""
    if drop_prob <= 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep_prob)
    return x * mask / keep_prob


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


def default_norm_factory(dim: int) -> nn.Module:
    return nn.LayerNorm(dim)


def rms_norm_factory(dim: int) -> nn.Module:
    return RMSNorm(dim)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * norm * self.weight


class FramePatchEmbed(nn.Module):
    """Tokenize each video frame independently with a 2D patch convolution.

    Input ``(B, T, C, H, W)`` -> tokens ``(B, T, S, D)`` where ``S`` is the
    number of spatial patches per frame. This is the simplest, most readable
    video tokenizer and the AeroJEPA default.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_spatial = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.proj(x)  # (B*T, D, gh, gw)
        x = x.flatten(2).transpose(1, 2)  # (B*T, S, D)
        return x.reshape(b, t, self.num_spatial, -1)


class TubeletEmbed(nn.Module):
    """Tokenize short space-time "tubelets" with a 3D convolution.

    Input ``(B, T, C, H, W)`` -> tokens ``(B, T', S, D)`` with ``T' =
    T // tubelet_size``. Tubelets bake a little motion into each token before
    attention ever runs; enable via ``tokenizer: tubelet`` in the config.
    """

    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_chans: int = 3,
        embed_dim: int = 192,
        tubelet_size: int = 2,
    ) -> None:
        super().__init__()
        self.grid_size = img_size // patch_size
        self.num_spatial = self.grid_size ** 2
        self.tubelet_size = tubelet_size
        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (B, C, T, H, W)
        x = self.proj(x)  # (B, D, T', gh, gw)
        b, d, tp, gh, gw = x.shape
        x = x.reshape(b, d, tp, gh * gw).permute(0, 2, 3, 1)  # (B, T', S, D)
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4)
        act = nn.GELU() if activation == "gelu" else nn.SiLU()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        hidden_dim = int(math.ceil(hidden_dim / 64) * 64)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        x = self.w_down(gate * up)
        return self.drop(x)


class Attention(nn.Module):
    """Multi-head self-attention that can optionally cache its attention map.

    The cached map (``last_attn``) powers the "where does the predictor look,
    loop by loop" visualizations that made the parent project interpretable.
    Caching is off by default so it never slows down training.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 3,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.store_attn = False
        self.last_attn: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if self.store_attn:
            self.last_attn = attn.detach()
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """ViT block with swappable norm and optional sandwich (pre + post) norms.

    Sandwich normalization -- an extra norm *after* each sub-layer, inside the
    residual path -- was the single most important stabilizer for the shared
    weight recurrent predictor in the parent project, so it is preserved here.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_factory: Callable[[int], nn.Module] = default_norm_factory,
        sandwich_norm: bool = False,
        ffn_type: str = "mlp",
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.sandwich_norm = sandwich_norm
        hidden_dim = int(dim * mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm1 = norm_factory(dim)
        self.norm2 = norm_factory(dim)
        self.norm_attn_out = norm_factory(dim) if sandwich_norm else None
        self.norm_ffn_out = norm_factory(dim) if sandwich_norm else None

        self.attn = Attention(dim, num_heads=num_heads, proj_drop=dropout)
        if ffn_type == "swiglu":
            self.mlp: nn.Module = SwiGLU(dim, hidden_dim=hidden_dim, dropout=dropout)
        else:
            self.mlp = Mlp(dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.norm_attn_out is not None:
            x = self.norm_attn_out(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.norm_ffn_out is not None:
            x = self.norm_ffn_out(x)
        return x


class BlockStack(nn.Module):
    """Reusable stack of transformer blocks (re-applied by the looped predictor)."""

    def __init__(self, blocks: nn.ModuleList) -> None:
        super().__init__()
        self.blocks = blocks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


def build_pos_embed(num_tokens: int, embed_dim: int) -> nn.Parameter:
    return nn.Parameter(torch.zeros(1, num_tokens, embed_dim))


def init_pos_embed(pos_embed: nn.Parameter) -> None:
    nn.init.trunc_normal_(pos_embed, std=0.02)


def spatiotemporal_pos_embed(
    spatial_pos: torch.Tensor,
    temporal_pos: torch.Tensor,
) -> torch.Tensor:
    """Combine spatial and temporal position tables into one per-token table.

    ``spatial_pos`` is ``(1, S, D)`` and ``temporal_pos`` is ``(1, T, D)``. We
    add them (a factorized space-time positional encoding) and flatten so that
    token index ``t * S + s`` carries frame ``t`` and patch ``s``. Factorizing
    keeps the parameter count tiny compared with a full ``(T*S, D)`` table.
    """
    t = temporal_pos.shape[1]
    s = spatial_pos.shape[1]
    d = spatial_pos.shape[2]
    combined = temporal_pos.view(1, t, 1, d) + spatial_pos.view(1, 1, s, d)
    return combined.reshape(1, t * s, d)
