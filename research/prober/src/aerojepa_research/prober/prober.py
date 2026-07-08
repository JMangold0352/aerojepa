"""Prober modules: structured physics prober + plain MLP ablation head.

The structured prober (``Prober``) maps a latent rollout from the frozen
predictor (+ actions) to residual accelerations consumed by the kinematic
integrator. The plain MLP head (``PlainMLPHead``) maps the same inputs directly
to metric state -- the no-physics ablation arm.

Input latent convention: the frozen AeroJEPA predictor outputs target latents of
shape (B, n_tgt, encoder_dim=192). For a future-frame world model n_tgt equals
num_spatial (64) per predicted frame. We pool each frame's spatial tokens with a
learned projection to get one (B, T, latent_dim) sequence before the prober MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Latent dimension of the frozen encoder (confirmed from checkpoints).
ENCODER_DIM = 192
# AeroJEPA 6-DoF action dimension.
ACTION_DIM = 6


class Prober(nn.Module):
    """Structured physics prober: latent rollout + actions -> residual accelerations.

    Outputs residual linear acceleration (3) and residual angular acceleration (3),
    which are fed to :class:`KinematicIntegrator`. Keeping the output as a residual
    (not a full acceleration) lets the nominal physics model carry the bulk of the
    dynamics and lets the prober focus on what the nominal model gets wrong.

    Parameters
    ----------
    latent_dim : int
        Per-frame latent dimension after pooling (default 192).
    action_dim : int
        Action dimension (default 6).
    hidden_dim : int
        Hidden width of the MLP.
    num_layers : int
        Number of MLP layers (input -> hidden -> ... -> output).
    """

    def __init__(
        self,
        latent_dim: int = ENCODER_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 24,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        in_dim = latent_dim + action_dim
        out_dim = 6  # 3 residual linear accel + 3 residual angular accel

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.mlp = nn.Sequential(*layers)

        # Init: small outputs so the prober starts near "trust the nominal model".
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Scale down the final layer so initial residuals are tiny.
        last = self.mlp[-1]
        with torch.no_grad():
            last.weight.mul_(0.01)

    def forward(self, latent: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict residual accelerations for each frame.

        Parameters
        ----------
        latent : (B, T, latent_dim)  per-frame pooled latent rollout
        actions : (B, T, action_dim)

        Returns
        -------
        res_lin : (B, T, 3)
        res_ang : (B, T, 3)
        """
        x = torch.cat([latent, actions], dim=-1)
        out = self.mlp(x)
        res_lin, res_ang = torch.split(out, [3, 3], dim=-1)
        return res_lin, res_ang

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PlainMLPHead(nn.Module):
    """Ablation arm: latent rollout + actions -> direct metric state (no integrator).

    This is the "plain MLP head" the charter ablates against. It predicts the
    full 12-D metric state per frame directly from latents + actions, with no
    kinematic structure and no integrator. Same input/hidden budget as the
    structured prober for a fair comparison.
    """

    def __init__(
        self,
        latent_dim: int = ENCODER_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = 24,
        num_layers: int = 2,
        state_dim: int = 12,
    ) -> None:
        super().__init__()
        in_dim = latent_dim + action_dim
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.GELU())
            d = hidden_dim
        layers.append(nn.Linear(d, state_dim))
        self.mlp = nn.Sequential(*layers)
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Predict (B, T, state_dim) metric states directly."""
        x = torch.cat([latent, actions], dim=-1)
        return self.mlp(x)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pool_latents(per_frame_latents: torch.Tensor) -> torch.Tensor:
    """Mean-pool spatial tokens within each predicted frame.

    The frozen predictor returns (B, T*n_spatial, encoder_dim) target latents
    for a rollout of T future frames (n_spatial tokens each). Pool to one vector
    per frame: (B, T, encoder_dim).

    Parameters
    ----------
    per_frame_latents : (B, T*n_spatial, encoder_dim) or (B, T, encoder_dim)
        If already pooled, returned unchanged.
    """
    if per_frame_latents.dim() == 3:
        # Already (B, T, D) -- assume caller pooled.
        return per_frame_latents
    if per_frame_latents.dim() != 3:
        raise ValueError(f"expected 3D latent tensor, got shape {tuple(per_frame_latents.shape)}")
    # Ambiguous 3D case: assume already pooled.
    return per_frame_latents
