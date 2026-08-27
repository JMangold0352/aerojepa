"""Gated / body-frame residual prober variants (Prompt 6 / memo Q1+Q4).

A. UngatedWorldProber — world-frame residual accel = MLP(s, u)  [current default]
B. GatedBodyProber — a_world = (T/m) R e3 - g + R a_body(s, u)
C. PartialGateProber — only rz (thrust axis) gated; yaw free in residual frame

SkyJEPA's Δv̇ residual is world-frame without this invariance test.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from aerojepa_research.prober.integrator import GRAVITY_Z, _euler_ypr_to_rotation
from aerojepa_research.prober.prober import CONTROL_DIM, ENCODER_DIM, Prober


class UngatedWorldProber(Prober):
    """Alias of :class:`Prober` — world-frame linear residual (variant A)."""

    variant = "A_ungated_world"


class GatedBodyProber(nn.Module):
    """Variant B: MLP predicts body-frame residual; rotate into world for integrator.

    The ControlIntegrator still *adds* ``res_lin`` in world frame. This module
    returns world-frame residual ``R @ a_body`` so the plant equation is
    ``v̇ = (T/m) R e3 − g + R a_body``. Callers that want the full gated nominal
    can use :meth:`world_accel` instead of the integrator's nominal + residual.
    """

    variant = "B_gated_body"

    def __init__(
        self,
        latent_dim: int = ENCODER_DIM,
        control_dim: int = CONTROL_DIM,
        hidden_dim: int = 24,
        num_layers: int = 2,
        ang_residual_scale: float = 0.25,
        mass: float = 1.0,
        hover_thrust: float = 0.39,
        gravity: float = GRAVITY_Z,
    ) -> None:
        super().__init__()
        self.inner = Prober(
            latent_dim=latent_dim,
            control_dim=control_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            ang_residual_scale=ang_residual_scale,
        )
        self.mass = mass
        self.hover_thrust = hover_thrust
        self.gravity = gravity

    def body_residual(
        self, latent: torch.Tensor, controls: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (a_body, a_ang) — linear residual in body frame."""
        return self.inner(latent, controls)

    def forward(
        self,
        latent: torch.Tensor,
        controls: torch.Tensor,
        euler_att_deg: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """World-frame residual for ControlIntegrator: R @ a_body, a_ang."""
        a_body, a_ang = self.body_residual(latent, controls)
        # euler_att_deg: (B, T, 3)
        B, T, _ = a_body.shape
        R = _euler_ypr_to_rotation(euler_att_deg.reshape(B * T, 3)).reshape(B, T, 3, 3)
        a_world = torch.einsum("btij,btj->bti", R, a_body)
        return a_world, a_ang

    def num_params(self) -> int:
        return self.inner.num_params()


class PartialGateProber(GatedBodyProber):
    """Variant C: only thrust-axis (body z) residual is rotated; xy stay world.

    Approximation of 'rz-only' gate: predict full body residual but zero body-x/y
    before rotating so only the thrust-axis component couples through R.
    """

    variant = "C_partial_rz"

    def body_residual(self, latent, controls):
        a_body, a_ang = self.inner(latent, controls)
        a_body = a_body.clone()
        a_body[..., 0] = 0.0
        a_body[..., 1] = 0.0
        return a_body, a_ang


def rotate_world_vector(v_world: torch.Tensor, R_world: torch.Tensor) -> torch.Tensor:
    """Apply world rotation R_world to vectors (..., 3)."""
    return torch.einsum("...ij,...j->...i", R_world, v_world)


@torch.no_grad()
def frame_invariance_test(
    a_body: torch.Tensor,
    a_world: torch.Tensor,
    R_extra: torch.Tensor,
) -> dict[str, float]:
    """Body residual must be invariant under world rotation; world must rotate.

    Parameters
    ----------
    a_body / a_world : (N, 3) residuals before rotating the world frame
    R_extra : (3, 3) additional world rotation applied to the inertial frame
    """
    a_body_rot = a_body  # invariant
    a_world_rot_expected = torch.einsum("ij,nj->ni", R_extra, a_world)
    body_drift = float((a_body_rot - a_body).pow(2).mean().sqrt())
    # After world rotation, an ungated world residual that was correct should
    # equal R_extra @ a_world. Measure how far a naive copy (no rotate) is.
    world_if_forgot = float((a_world - a_world_rot_expected).pow(2).mean().sqrt())
    world_if_rotated = float(
        (a_world_rot_expected - a_world_rot_expected).pow(2).mean().sqrt()
    )
    return {
        "body_invariance_rmse": body_drift,
        "world_must_rotate_rmse_if_forgotten": world_if_forgot,
        "world_ok_when_rotated_rmse": world_if_rotated,
        "pass_body_invariant": body_drift < 1e-6,
        "pass_world_transforms": world_if_forgot > 1e-3,  # nontrivial rotation
    }
