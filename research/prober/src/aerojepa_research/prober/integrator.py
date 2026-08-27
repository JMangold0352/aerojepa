"""Differentiable integrators for AeroProber metric rollouts.

**Use :class:`ControlIntegrator` for all leak-free / headline work.** It consumes
exogenous PyFlyt controls ``(vp, vq, vr, T)`` — body-rate setpoints (rad/s) plus
a *normalized* collective thrust — and advances
``(pos, vel, euler_att_deg, ang_vel_deg)`` with a first-order Euler step.

Plant facts (z-up, mass default 1 kg, ``hover_thrust=0.39``, ``dt=0.025``):
    a_lin_nom = R(euler) @ e3 * (T/hover)*|g| + g*e3
    a_ang_nom = (deg(vp,vq,vr) - omega) / dt
Residuals: world-frame linear + body-frame angular. Attitude is stored as an
Euler chart with wrap; body rates ``(p,q,r)`` are converted to Euler rates
before the chart step — **not** a Lie-group Exp update (see
``docs/CORRECTNESS.md`` V5 and ``research/prober/note.md``). Angular residuals
are typically scaled by 0.25 in ``Prober``.

:class:`KinematicIntegrator` is **legacy / leaky**: it treats AeroJEPA 6-DoF
pose-delta actions as velocity/attitude targets (can equal GT). Do not use it
for reported v3 numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Gravity in m/s^2, acting on the world z-velocity. Sign convention: z-up,
# so gravity is negative.
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
    ang_vel: torch.Tensor    # body (p, q, r) deg/s

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
    """LEGACY / LEAKY first-order integrator (state-derived AeroJEPA actions).

    Prefer :class:`ControlIntegrator` for all new work. Kept for provenance and
    comparison to the pre-leak-fix design.
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


def _euler_ypr_to_rotation(att_deg: torch.Tensor) -> torch.Tensor:
    """Rotation matrix (body -> world) from (yaw, pitch, roll) in degrees.

    Returns (..., 3, 3). Uses the standard ZYX (yaw-pitch-roll) convention.
    """
    y = torch.deg2rad(att_deg[..., 0])
    p = torch.deg2rad(att_deg[..., 1])
    r = torch.deg2rad(att_deg[..., 2])
    cy, sy = torch.cos(y), torch.sin(y)
    cp, sp = torch.cos(p), torch.sin(p)
    cr, sr = torch.cos(r), torch.sin(r)
    # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    R = torch.stack([
        cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr,
        sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr,
        -sp,     cp * sr,                cp * cr,
    ], dim=-1).reshape(*att_deg.shape[:-1], 3, 3)
    return R


def body_rates_to_euler_rates_deg(
    euler_att_ypr_deg: torch.Tensor, omega_pqr_deg: torch.Tensor
) -> torch.Tensor:
    """Map body rates (p,q,r) [deg/s] → Euler chart rates (ψ̇,θ̇,φ̇) [deg/s].

    ``euler_att_ypr_deg`` is (..., 3) yaw–pitch–roll. Near pitch ±90° the chart
    is singular — prefer Exp on SO(3) (see ``so3_integrators``) for those regimes.
    """
    ypr = torch.deg2rad(euler_att_ypr_deg)
    pitch = ypr[..., 1]
    roll = ypr[..., 2]
    p = torch.deg2rad(omega_pqr_deg[..., 0])
    q = torch.deg2rad(omega_pqr_deg[..., 1])
    r = torch.deg2rad(omega_pqr_deg[..., 2])
    sp, cp = torch.sin(pitch), torch.cos(pitch)
    sr, cr = torch.sin(roll), torch.cos(roll)
    cos_pitch = cp.sign() * cp.abs().clamp(min=1e-3)
    tan_pitch = sp / cos_pitch
    roll_dot = p + q * sr * tan_pitch + r * cr * tan_pitch
    pitch_dot = q * cr - r * sr
    yaw_dot = (q * sr + r * cr) / cos_pitch
    return torch.rad2deg(torch.stack([yaw_dot, pitch_dot, roll_dot], dim=-1))


class ControlIntegrator(torch.nn.Module):
    """Leak-free integrator: raw control commands -> metric state.

    Unlike :class:`KinematicIntegrator` (whose action contains state-derived
    velocities/attitude-deltas), this integrator consumes genuinely exogenous
    control commands ``(vp, vq, vr, T)`` -- angular-rate setpoints and a
    collective-thrust setpoint. Nothing in the action reveals the ground-truth
    state, so any metric accuracy the prober achieves must come from the latent.

    Nominal physics model (what the control alone does):
        - Thrust ``T`` is a *normalized* PyFlyt-style setpoint in roughly
          ``[0, 0.8]``. We map it to force so that ``T ≈ hover_thrust`` cancels
          gravity when level: ``force = (T / hover_thrust) * mass * |g|``.
        - Angular-rate setpoints ``(vp, vq, vr)`` (rad/s) drive the body-frame
          angular velocity via a first-order hold.

    The prober predicts residual linear and angular accelerations added to
    these nominal accelerations before the Euler step -- it learns what the
    simple nominal model gets wrong (drag, motor dynamics, coupling).

    Parameters
    ----------
    dt : float
        Per-frame time step [s].
    gravity : float
        World-z gravitational acceleration [m/s^2].
    mass : float
        Quadrotor mass [kg]. PyFlyt's QuadX default is ~1.0 kg.
    hover_thrust : float
        Normalized thrust setpoint that should hover when level.
    """

    def __init__(
        self,
        dt: float = 0.025,
        gravity: float = GRAVITY_Z,
        mass: float = 1.0,
        hover_thrust: float = 0.39,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.gravity = gravity
        self.mass = mass
        self.hover_thrust = hover_thrust

    def nominal_accel(
        self, control: torch.Tensor, state: MetricState
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Nominal linear and angular accelerations from control commands.

        Parameters
        ----------
        control : (B, 4)  [vp, vq, vr, T]  (rad/s, rad/s, rad/s, thrust proxy)
        state : current MetricState (attitude in degrees, ang_vel in deg/s)

        Returns
        -------
        a_lin_nom : (B, 3)  world-frame linear acceleration [m/s^2]
        a_ang_nom : (B, 3)  body-frame angular acceleration [deg/s^2]
        """
        dt = self.dt
        T = control[..., 3]  # normalized thrust setpoint
        # Map normalized thrust so hover_thrust cancels |gravity| when level.
        hover = max(float(self.hover_thrust), 1e-6)
        thrust_acc = (T / hover) * abs(float(self.gravity))  # m/s^2 along body +z
        R = _euler_ypr_to_rotation(state.euler_att)  # (B, 3, 3)
        body_thrust = torch.zeros(*T.shape, 3, device=T.device, dtype=T.dtype)
        body_thrust[..., 2] = thrust_acc
        a_lin_nom = torch.einsum("...ij,...j->...i", R, body_thrust)  # (B, 3)
        # Add gravity (world z).
        a_lin_nom = a_lin_nom.clone()
        a_lin_nom[..., 2] = a_lin_nom[..., 2] + self.gravity

        # Angular-rate setpoint -> angular acceleration (first-order hold).
        # control[:3] are in rad/s; state.ang_vel is in deg/s. Convert.
        setpoint_deg = torch.rad2deg(control[..., :3])  # (B, 3) deg/s
        a_ang_nom = (setpoint_deg - state.ang_vel) / dt  # deg/s^2
        return a_lin_nom, a_ang_nom

    def step(
        self,
        state: MetricState,
        control: torch.Tensor,
        res_lin: torch.Tensor,
        res_ang: torch.Tensor,
    ) -> MetricState:
        """Advance the state by one frame given a control command + residuals."""
        dt = self.dt
        a_lin_nom, a_ang_nom = self.nominal_accel(control, state)
        a_lin = a_lin_nom + res_lin
        a_ang = a_ang_nom + res_ang

        new_vel = state.vel + a_lin * dt
        new_pos = state.pos + new_vel * dt
        new_ang_vel = state.ang_vel + a_ang * dt  # body (p,q,r) deg/s
        # Chart update: body rates → Euler (yaw,pitch,roll) rates, then wrap.
        euler_rates = body_rates_to_euler_rates_deg(state.euler_att, new_ang_vel)
        new_att = wrap_degrees(state.euler_att + euler_rates * dt)
        return MetricState(pos=new_pos, vel=new_vel, euler_att=new_att, ang_vel=new_ang_vel)

    def rollout(
        self,
        init_state: MetricState,
        controls: torch.Tensor,
        res_lin: torch.Tensor,
        res_ang: torch.Tensor,
    ) -> MetricState:
        """Roll the state forward over a sequence of controls + residuals.

        Parameters
        ----------
        init_state : MetricState
        controls : (B, T, 4)  raw [vp, vq, vr, T]
        res_lin : (B, T, 3)
        res_ang : (B, T, 3)
        """
        T = controls.shape[1]
        if res_lin.shape[1] != T or res_ang.shape[1] != T:
            raise ValueError(
                f"residual horizon {res_lin.shape[1]}/{res_ang.shape[1]} != control horizon {T}"
            )
        state = init_state.clone()
        pos_list, vel_list, att_list, av_list = [], [], [], []
        for t in range(T):
            state = self.step(state, controls[:, t], res_lin[:, t], res_ang[:, t])
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
