"""Differentiable kinematic integrator for a quadrotor (Euler-angle attitude).

The integrator advances a metric state ``(pos, vel, euler_att, ang_vel)`` forward
in time given an action (the AeroJEPA 6-DoF pose-delta convention) and a residual
acceleration predicted by the prober.

State layout (all in metric units, degrees for attitude):
    pos       (..., 3)   world-frame position [m]
    vel       (..., 3)   world-frame velocity [m/s]
    euler_att (..., 3)   (yaw, pitch, roll) in degrees, wrapped to (-180, 180]
    ang_vel   (..., 3)   body-frame angular velocity [deg/s]

Action layout (AeroJEPA ACTION_COLUMNS):
    (dx, dy, d_altitude, d_yaw, d_pitch, d_roll)
    -- per-frame pose deltas. The first three are body velocities (m/s proxy),
       the last three are wrapped frame-to-frame attitude deltas (deg).

Residual layout (prober output):
    res_lin (..., 3)   residual linear acceleration [m/s^2]
    res_ang (..., 3)   residual angular acceleration [deg/s^2]

Nominal physics (what the action alone would do):
    - The body-velocity channels of the action are taken as the nominal
      velocity update (a first-order hold: the action *is* the velocity).
    - The attitude-delta channels of the action are taken as the nominal
      angular velocity update.
    Gravity is applied as a constant downward acceleration on the world-frame
    z-velocity so that an action of all zeros does not hover -- the prober must
    learn to cancel it. This keeps the nominal model honest and gives the
    residual a real job to do.

Integration: first-order Euler with the action's per-frame dt. Attitude is
wrapped with ``wrap_degrees`` after each step so it stays in (-180, 180].
Everything is differentiable through standard torch ops.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Gravity in m/s^2, acting on the world z-velocity. Sign convention: z-up,
# so gravity is negative. ``d_altitude`` in the action convention is ``vgz``
# (body z-velocity), which means an action of zero should make the drone
# fall; the prober learns the residual to counteract this.
GRAVITY_Z = -9.81


def wrap_degrees(delta: torch.Tensor) -> torch.Tensor:
    """Wrap angle differences to (-180, 180] (tensor version of telemetry helper)."""
    return (delta + 180.0) % 360.0 - 180.0


@dataclass
class MetricState:
    """A batched metric state. Tensors are ``(B, 3)`` shaped."""

    pos: torch.Tensor
    vel: torch.Tensor
    euler_att: torch.Tensor  # (yaw, pitch, roll), degrees
    ang_vel: torch.Tensor    # (deg/s)

    @classmethod
    def zeros(cls, batch_size: int, device: torch.device | str = "cpu") -> MetricState:
        z = torch.zeros(batch_size, 3, device=device)
        return cls(pos=z.clone(), vel=z.clone(), euler_att=z.clone(), ang_vel=z.clone())

    def stack(self) -> torch.Tensor:
        """Concatenate to (B, 12): [pos, vel, euler_att, ang_vel]."""
        return torch.cat([self.pos, self.vel, self.euler_att, self.ang_vel], dim=-1)

    @classmethod
    def from_stack(cls, stacked: torch.Tensor) -> MetricState:
        """Inverse of :meth:`stack`."""
        pos, vel, att, av = torch.split(stacked, [3, 3, 3, 3], dim=-1)
        return cls(pos=pos, vel=vel, euler_att=att, ang_vel=av)

    def clone(self) -> MetricState:
        return MetricState(
            pos=self.pos.clone(),
            vel=self.vel.clone(),
            euler_att=self.euler_att.clone(),
            ang_vel=self.ang_vel.clone(),
        )


class KinematicIntegrator(torch.nn.Module):
    """First-order differentiable integrator for the prober rollout.

    The nominal model derives velocity/angular-velocity updates directly from
    the AeroJEPA action convention (body velocities + attitude deltas), plus
    gravity on world-z. The prober's residual accelerations are added to the
    derived accelerations before the Euler step.

    Parameters
    ----------
    dt : float
        Per-frame time step [s]. Defaults to 1/15 (the Parrot Wilds framerate).
    gravity : float
        World-z gravitational acceleration [m/s^2]. Set to 0.0 to disable.
    """

    def __init__(self, dt: float = 1.0 / 15.0, gravity: float = GRAVITY_Z) -> None:
        super().__init__()
        self.dt = dt
        self.gravity = gravity

    def nominal_accel(self, action: torch.Tensor, state: MetricState) -> tuple[torch.Tensor, torch.Tensor]:
        """Nominal linear and angular accelerations implied by the action.

        The action's body-velocity channels (``dx, dy, d_altitude``) are a
        first-order hold on velocity -- i.e. they carry the per-frame velocity
        target (matching ``telemetry.ACTION_COLUMNS`` where ``dx=vgx`` etc.).
        The implied linear acceleration is therefore
        ``(action_vel - current_vel) / dt``. The attitude-delta channels
        likewise imply an angular acceleration via the same first-order hold.

        Parameters
        ----------
        action : (B, 6)  [dx, dy, d_altitude, d_yaw, d_pitch, d_roll]
        state : current MetricState

        Returns
        -------
        a_lin_nom : (B, 3)  nominal world-frame linear acceleration
        a_ang_nom : (B, 3)  nominal body-frame angular acceleration [deg/s^2]
        """
        dt = self.dt
        # Nominal velocity target from action's body-velocity channels.
        action_vel = action[..., :3]
        a_lin_nom = (action_vel - state.vel) / dt
        # Gravity acts on world z (third channel).
        a_lin_nom = a_lin_nom.clone()
        a_lin_nom[..., 2] = a_lin_nom[..., 2] + self.gravity

        # Nominal angular-velocity target from action's attitude-delta channels.
        action_ang = action[..., 3:6]
        a_ang_nom = (action_ang - state.ang_vel) / dt
        return a_lin_nom, a_ang_nom

    def step(
        self,
        state: MetricState,
        action: torch.Tensor,
        res_lin: torch.Tensor,
        res_ang: torch.Tensor,
    ) -> MetricState:
        """Advance the state by one frame.

        Parameters
        ----------
        state : MetricState (B, ...)
        action : (B, 6)
        res_lin : (B, 3)  residual linear acceleration from the prober
        res_ang : (B, 3)  residual angular acceleration from the prober

        Returns
        -------
        next_state : MetricState
        """
        dt = self.dt
        a_lin_nom, a_ang_nom = self.nominal_accel(action, state)
        a_lin = a_lin_nom + res_lin
        a_ang = a_ang_nom + res_ang

        new_vel = state.vel + a_lin * dt
        new_pos = state.pos + new_vel * dt
        new_ang_vel = state.ang_vel + a_ang * dt
        new_att = wrap_degrees(state.euler_att + new_ang_vel * dt)

        return MetricState(pos=new_pos, vel=new_vel, euler_att=new_att, ang_vel=new_ang_vel)

    def rollout(
        self,
        init_state: MetricState,
        actions: torch.Tensor,
        res_lin: torch.Tensor,
        res_ang: torch.Tensor,
    ) -> MetricState:
        """Roll the state forward over a sequence of actions + residuals.

        Parameters
        ----------
        init_state : MetricState  (B, ...)
        actions : (B, T, 6)
        res_lin : (B, T, 3)  per-frame residual linear acceleration
        res_ang : (B, T, 3)  per-frame residual angular acceleration

        Returns
        -------
        traj : MetricState whose tensors are (B, T, ...) -- the state *after*
            each step (excludes the initial state).
        """
        T = actions.shape[1]
        if res_lin.shape[1] != T or res_ang.shape[1] != T:
            raise ValueError(
                f"residual horizon {res_lin.shape[1]}/{res_ang.shape[1]} != action horizon {T}"
            )

        state = init_state.clone()
        pos_list, vel_list, att_list, av_list = [], [], [], []
        for t in range(T):
            state = self.step(state, actions[:, t], res_lin[:, t], res_ang[:, t])
            pos_list.append(state.pos)
            vel_list.append(state.vel)
            att_list.append(state.euler_att)
            av_list.append(state.ang_vel)

        return MetricState(
            pos=torch.stack(pos_list, dim=1),
            vel=torch.stack(vel_list, dim=1),
            euler_att=torch.stack(att_list, dim=1),
            ang_vel=torch.stack(av_list, dim=1),
        )
