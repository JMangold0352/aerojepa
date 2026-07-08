"""Supervised multi-horizon loss for the prober.

The structured prober arm integrates residual accelerations into a metric
trajectory and computes MSE against ground truth over all horizons. The plain
MLP arm computes MSE directly on its per-frame state predictions.

Attitude is special: Euler angles wrap at +-180, so a raw MSE on attitude would
penalize a 359->1 prediction as a 358-degree error instead of 2 degrees. We use
a wrapped-angle error for the attitude channels.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aerojepa_research.prober.integrator import KinematicIntegrator, MetricState
from aerojepa_research.prober.prober import PlainMLPHead, Prober


def wrapped_angle_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Shortest-angle error between predicted and target Euler angles (degrees).

    Returns the signed error in (-180, 180], so squaring it gives the true
    squared angular distance.
    """
    return ((pred - target + 180.0) % 360.0) - 180.0


def metric_state_mse(pred_stack: torch.Tensor, gt_stack: torch.Tensor) -> torch.Tensor:
    """MSE over a (B, T, 12) metric-state trajectory with wrapped attitude.

    Layout: [pos(3), vel(3), euler_att_deg(3), ang_vel(3)].
    """
    # Position, velocity, angular velocity: plain squared error.
    lin_err = pred_stack[..., 0:6] - gt_stack[..., 0:6]
    ang_vel_err = pred_stack[..., 9:12] - gt_stack[..., 9:12]
    # Attitude: wrapped shortest-angle error.
    att_err = wrapped_angle_error(pred_stack[..., 6:9], gt_stack[..., 6:9])
    sq_err = torch.cat([lin_err, att_err, ang_vel_err], dim=-1) ** 2
    return sq_err.mean()


class StructuredProberLoss(nn.Module):
    """Loss for the structured physics prober: integrate -> MSE vs GT.

    The loss sums MSE over position, velocity, wrapped attitude, and angular
    velocity across all predicted horizons.
    """

    def __init__(self, integrator: KinematicIntegrator) -> None:
        super().__init__()
        self.integrator = integrator

    def forward(
        self,
        prober: Prober,
        latents: torch.Tensor,
        actions: torch.Tensor,
        init_state: MetricState,
        gt_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the supervised rollout loss.

        Returns
        -------
        loss : scalar tensor
        pred_stack : (B, T, 12) predicted metric trajectory (for logging)
        """
        res_lin, res_ang = prober(latents, actions)
        pred_traj = self.integrator.rollout(init_state, actions, res_lin, res_ang)
        pred_stack = pred_traj.stack()
        loss = metric_state_mse(pred_stack, gt_states)
        return loss, pred_stack


class PlainMLPLoss(nn.Module):
    """Loss for the plain MLP ablation arm: direct state prediction -> MSE.

    The plain head predicts each frame's metric state directly from latents +
    actions, with no integrator. We still apply wrapped-angle MSE on attitude.
    """

    def forward(
        self,
        head: PlainMLPHead,
        latents: torch.Tensor,
        actions: torch.Tensor,
        init_state: MetricState,
        gt_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_stack = head(latents, actions)
        loss = metric_state_mse(pred_stack, gt_states)
        return loss, pred_stack


class NaiveLatentLoss(nn.Module):
    """Loss for the no-prober ablation arm: raw latent -> linear projection.

    The "no prober" baseline maps the frozen latent directly to metric state via
    a single linear layer (the simplest possible decoder). This tests whether
    the metric state is trivially linearly decodable from the latent.
    """

    def __init__(self, latent_dim: int = 192, state_dim: int = 12) -> None:
        super().__init__()
        self.proj = nn.Linear(latent_dim, state_dim)

    def forward(
        self,
        latents: torch.Tensor,
        actions: torch.Tensor,
        init_state: MetricState,
        gt_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_stack = self.proj(latents)
        loss = metric_state_mse(pred_stack, gt_states)
        return loss, pred_stack

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
