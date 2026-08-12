"""Leak-free helpers for Wilds real-data evaluation.

Two independent fixes for the sim-to-real gap:

1. **Body → world velocity.** Parrot logs store body-frame velocities; the
   ControlIntegrator predicts world-frame velocity. Comparing them raw inflates
   Wilds velocity RMSE. Rotate GT body velocity by the GT attitude before
   scoring (geometry only — zero parameters, no action leak).

2. **Eval control priors.** Parrot logs lack motor commands. Options:
   - ``zeros``: gravity-only nominal (original stress test).
   - ``hover``: constant thrust prior matching the PyFlyt hover setpoint
     (~0.39), rates zero. Exogenous constant — not derived from GT velocity
     or pose-delta actions (avoids the v2 leak pattern).

IMU rate setpoints are intentionally *not* offered as a default: feeding
``ang_vel`` as rate setpoints while initializing from GT state collapses the
angular residual and reintroduces an attitude shortcut.
"""

from __future__ import annotations

import torch

from aerojepa_research.prober.integrator import _euler_ypr_to_rotation

# PyFlyt QuadX hover-region thrust setpoint used in closed-loop demos / data.
DEFAULT_HOVER_THRUST = 0.39


def body_vel_to_world(vel_body: torch.Tensor, att_deg: torch.Tensor) -> torch.Tensor:
    """Rotate body-frame velocity to world frame via yaw-pitch-roll.

    Parameters
    ----------
    vel_body : (..., 3)
    att_deg : (..., 3)  (yaw, pitch, roll) degrees

    Returns
    -------
    vel_world : (..., 3)
    """
    R = _euler_ypr_to_rotation(att_deg)  # (..., 3, 3)
    return torch.einsum("...ij,...j->...i", R, vel_body)


def metric_stack_body_vel_to_world(stack: torch.Tensor) -> torch.Tensor:
    """Return a copy of a (..., 12) metric stack with vel rotated body→world."""
    out = stack.clone()
    out[..., 3:6] = body_vel_to_world(stack[..., 3:6], stack[..., 6:9])
    return out


def make_eval_controls(
    num_frames: int,
    *,
    mode: str = "zeros",
    hover_thrust: float = DEFAULT_HOVER_THRUST,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build (num_frames, 4) control tensor for Wilds eval.

    ``mode``:
      * ``zeros`` — no motor prior (original protocol).
      * ``hover`` — constant ``(0,0,0,hover_thrust)`` exogenous prior.
    """
    controls = torch.zeros(num_frames, 4, device=device, dtype=dtype)
    if mode == "zeros":
        return controls
    if mode == "hover":
        controls[:, 3] = hover_thrust
        return controls
    raise ValueError(f"unknown eval control mode {mode!r}; expected 'zeros' or 'hover'")
