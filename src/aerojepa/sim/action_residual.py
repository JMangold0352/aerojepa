"""Tiny learned residual on top of the heuristic AeroJEPA → PyFlyt action map.

The frozen :class:`~aerojepa.sim.planner.LatentPlanner` still imagines and scores
plans in AeroJEPA's 6-DoF action space. At execution time we convert those
actions with the hand-crafted :func:`aerojepa_to_pyflyt` map, then add a small
MLP residual conditioned on the imagined latent + the heuristic control:

    u = clip( heuristic(a_6) + residual(latent, heuristic, a_6) )

Only the residual head is trained; the world model and latent rollout stay frozen.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Keep in sync with ``closed_loop.DEFAULT_HOVER_THRUST`` (avoid circular import).
DEFAULT_HOVER_THRUST = 0.39
CONTROL_DIM = 4
AERO_DIM = 6


class ActionResidualHead(nn.Module):
    """Maps (pooled latent, heuristic PyFlyt control, AeroJEPA action) → Δcontrol.

    Kept deliberately tiny (~1–3k params) so it is a residual corrector, not a
    second controller.
    """

    def __init__(
        self,
        latent_dim: int = 192,
        hidden_dim: int = 16,
        aero_dim: int = AERO_DIM,
        control_dim: int = CONTROL_DIM,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.aero_dim = aero_dim
        self.control_dim = control_dim
        # Soft residual caps keep the head a corrector, not a second controller.
        self.max_rate_delta = 0.15
        self.max_thrust_delta = 0.02
        # Fraction of the latent-only residual kept after bias cancel.
        # 0 → pure relative-to-null-aero (safe hover); >0 lets latents fight wind
        # even when the planner's AeroJEPA action is near zero.
        self.latent_residual_gain = 0.35
        self.latent_norm = nn.LayerNorm(latent_dim)
        in_dim = latent_dim + control_dim + aero_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, control_dim),
        )
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.mlp[-1].weight.mul_(0.01)

    def forward(
        self,
        latent: torch.Tensor,
        heuristic: torch.Tensor,
        aero_action: torch.Tensor,
    ) -> torch.Tensor:
        """Predict control residual.

        Parameters
        ----------
        latent : (..., latent_dim)
        heuristic : (..., 4)  hand-crafted PyFlyt map
        aero_action : (..., 6)  AeroJEPA 6-DoF action

        Returns
        -------
        delta : (..., 4)
        """
        z = self.latent_norm(latent)
        delta = self._raw_delta(z, heuristic, aero_action)
        delta0 = self._raw_delta(z, heuristic, torch.zeros_like(aero_action))
        # Keep a slice of the latent-only correction for disturbance rejection
        # (wind), while still canceling most of the constant rate bias.
        gain = float(self.latent_residual_gain)
        return delta - (1.0 - gain) * delta0

    def _raw_delta(
        self,
        z: torch.Tensor,
        heuristic: torch.Tensor,
        aero_action: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z, heuristic, aero_action], dim=-1)
        raw = self.mlp(x)
        rates = torch.tanh(raw[..., :3]) * self.max_rate_delta
        thrust = torch.tanh(raw[..., 3:4]) * self.max_thrust_delta
        return torch.cat([rates, thrust], dim=-1)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def aerojepa_to_pyflyt_torch(
    action_6: torch.Tensor,
    *,
    hover_thrust: float = DEFAULT_HOVER_THRUST,
    rate_scale: float = 2.0,
    xy_scale: float = 3.5,
    alt_scale: float = 0.6,
) -> torch.Tensor:
    """Batched differentiable-friendly heuristic map (same signs as NumPy version)."""
    a = action_6
    if a.shape[-1] != 6:
        raise ValueError(f"expected last dim 6, got {tuple(a.shape)}")
    dx, dy, d_alt, d_yaw, d_pitch, d_roll = a.unbind(dim=-1)
    pi = torch.tensor(np.pi, device=a.device, dtype=a.dtype)
    vp = torch.clamp((-dy * xy_scale) + (d_roll * rate_scale), -pi, pi)
    vq = torch.clamp((dx * xy_scale) + (d_pitch * rate_scale), -pi, pi)
    vr = torch.clamp(d_yaw * rate_scale, -pi, pi)
    thrust = torch.clamp(hover_thrust + alt_scale * d_alt, 0.0, 0.8)
    return torch.stack([vp, vq, vr, thrust], dim=-1)


def apply_residual_control(
    aero_action: torch.Tensor,
    latent: torch.Tensor,
    residual_head: ActionResidualHead,
    *,
    hover_thrust: float = DEFAULT_HOVER_THRUST,
) -> torch.Tensor:
    """heuristic(a) + residual → clipped PyFlyt control ``(vp,vq,vr,T)``."""
    heur = aerojepa_to_pyflyt_torch(aero_action, hover_thrust=hover_thrust)
    delta = residual_head(latent, heur, aero_action)
    ctrl = heur + delta
    pi = torch.tensor(np.pi, device=ctrl.device, dtype=ctrl.dtype)
    ctrl = torch.stack(
        [
            torch.clamp(ctrl[..., 0], -pi, pi),
            torch.clamp(ctrl[..., 1], -pi, pi),
            torch.clamp(ctrl[..., 2], -pi, pi),
            torch.clamp(ctrl[..., 3], 0.0, 0.8),
        ],
        dim=-1,
    )
    return ctrl


def pool_plan_latents(pred_latents: torch.Tensor) -> torch.Tensor:
    """Mean-pool spatial tokens: ``(N, H, S, D) → (N, H, D)`` or ``(H, S, D) → (H, D)``."""
    if pred_latents.dim() == 4:
        return pred_latents.mean(dim=-2)
    if pred_latents.dim() == 3:
        return pred_latents.mean(dim=-2)
    raise ValueError(f"expected 3D or 4D latents, got shape {tuple(pred_latents.shape)}")


def residual_loss(
    residual_head: ActionResidualHead,
    latents: torch.Tensor,
    aero_actions: torch.Tensor,
    gt_controls: torch.Tensor,
    *,
    hover_thrust: float = DEFAULT_HOVER_THRUST,
    residual_l2: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervised loss: match GT PyFlyt controls, keep residual small.

    Parameters
    ----------
    latents : (B, T, D)
    aero_actions : (B, T, 6)
    gt_controls : (B, T, 4)
    """
    heur = aerojepa_to_pyflyt_torch(aero_actions, hover_thrust=hover_thrust)
    delta = residual_head(latents, heur, aero_actions)
    pred = heur + delta
    pi = torch.tensor(np.pi, device=pred.device, dtype=pred.dtype)
    pred = torch.stack(
        [
            torch.clamp(pred[..., 0], -pi, pi),
            torch.clamp(pred[..., 1], -pi, pi),
            torch.clamp(pred[..., 2], -pi, pi),
            torch.clamp(pred[..., 3], 0.0, 0.8),
        ],
        dim=-1,
    )
    mse = torch.mean((pred - gt_controls) ** 2)
    heur_mse = torch.mean((heur - gt_controls) ** 2)
    reg = torch.mean(delta ** 2)
    loss = mse + residual_l2 * reg
    stats = {
        "loss": float(loss.detach()),
        "mse": float(mse.detach()),
        "mse_gt": float(mse.detach()),
        "heur_mse": float(heur_mse.detach()),
        "delta_l2": float(reg.detach()),
    }
    return loss, stats


def load_residual_head(
    path: str | Path,
    device: torch.device,
    *,
    latent_dim: int | None = None,
) -> ActionResidualHead:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config") or {}
    head = ActionResidualHead(
        latent_dim=int(latent_dim or cfg.get("latent_dim", 192)),
        hidden_dim=int(cfg.get("hidden_dim", 16)),
    ).to(device)
    head.load_state_dict(ckpt["model"])
    head.eval()
    return head


def save_residual_checkpoint(
    path: str | Path,
    head: ActionResidualHead,
    *,
    config: dict[str, Any],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": head.state_dict(),
            "config": config,
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def map_aero_with_optional_residual(
    aero_action_6: np.ndarray | torch.Tensor,
    *,
    residual_head: ActionResidualHead | None = None,
    latent: torch.Tensor | None = None,
    hover_thrust: float = DEFAULT_HOVER_THRUST,
) -> np.ndarray:
    """NumPy-facing helper used by the closed-loop runner."""
    a = torch.as_tensor(aero_action_6, dtype=torch.float32).reshape(6)
    if residual_head is None or latent is None:
        return (
            aerojepa_to_pyflyt_torch(a, hover_thrust=hover_thrust)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
    residual_head.eval()
    with torch.no_grad():
        lat = latent.detach().float().reshape(-1)
        if lat.numel() != residual_head.latent_dim:
            # Allow (S, D) spatial tokens by mean-pooling.
            lat = lat.view(-1, residual_head.latent_dim).mean(dim=0)
        device = next(residual_head.parameters()).device
        ctrl = apply_residual_control(
            a.to(device),
            lat.to(device),
            residual_head,
            hover_thrust=hover_thrust,
        )
    return ctrl.detach().cpu().numpy().astype(np.float32)
