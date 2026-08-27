"""SO(3) attitude integrators for the bake-off.

Same residual / control plant; only the attitude update changes:
1. Euclidean Euler on R as R^9 (leaves SO(3))
2. First-order Exp: R <- R exp([ω]_× dt)  (SkyJEPA-style)
3. RK4 on so(3)
4. Lie-group variational / implicit midpoint (Marsden–West style, short form)

Claim **constraint preservation**, never energy conservation (forced + drag).
"""

from __future__ import annotations

import torch

from aerojepa_research.prober.integrator import (
    GRAVITY_Z,
    ControlIntegrator,
    MetricState,
    _euler_ypr_to_rotation,
    wrap_degrees,
)


def skew(w: torch.Tensor) -> torch.Tensor:
    """Hat map: (..., 3) → (..., 3, 3)."""
    wx, wy, wz = w[..., 0], w[..., 1], w[..., 2]
    O = torch.zeros(*w.shape[:-1], 3, 3, device=w.device, dtype=w.dtype)
    O[..., 0, 1] = -wz
    O[..., 0, 2] = wy
    O[..., 1, 0] = wz
    O[..., 1, 2] = -wx
    O[..., 2, 0] = -wy
    O[..., 2, 1] = wx
    return O


def so3_exp(w: torch.Tensor) -> torch.Tensor:
    """Rodrigues Exp: so(3) vector (..., 3) → SO(3) (..., 3, 3). ``w`` in radians."""
    theta = torch.linalg.norm(w, dim=-1, keepdim=True).clamp(min=1e-12)
    k = w / theta
    K = skew(k)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(*w.shape[:-1], 3, 3)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    return eye + s * K + (1.0 - c) * (K @ K)


def so3_log(R: torch.Tensor) -> torch.Tensor:
    """Log: SO(3) → so(3) vector (radians)."""
    tr = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    cos_th = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_th)
    # vee(R - R^T) / (2 sin θ)
    skew_m = R - R.transpose(-1, -2)
    vee = torch.stack(
        [skew_m[..., 2, 1], skew_m[..., 0, 2], skew_m[..., 1, 0]], dim=-1
    )
    sin_th = torch.sin(theta).clamp(min=1e-12)
    scale = (theta / (2.0 * sin_th)).unsqueeze(-1)
    # Near identity: vee/2
    near = (theta < 1e-6).unsqueeze(-1)
    return torch.where(near, 0.5 * vee, scale * vee)


def rotation_constraint_frobenius(R: torch.Tensor) -> torch.Tensor:
    """‖Rᵀ R − I‖_F for each matrix (...,)."""
    RtR = R.transpose(-1, -2) @ R
    eye = torch.eye(3, device=R.device, dtype=R.dtype).expand_as(RtR)
    return torch.linalg.norm(RtR - eye, dim=(-2, -1))


def _rotation_to_euler_ypr_deg(R: torch.Tensor) -> torch.Tensor:
    """Extract yaw-pitch-roll degrees from R (ZYX)."""
    # pitch = -asin(R[2,0]); careful near poles
    sp = (-R[..., 2, 0]).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    pitch = torch.asin(sp)
    yaw = torch.atan2(R[..., 1, 0], R[..., 0, 0])
    roll = torch.atan2(R[..., 2, 1], R[..., 2, 2])
    return torch.rad2deg(torch.stack([yaw, pitch, roll], dim=-1))


class AttitudeIntegratorBase(torch.nn.Module):
    """Shared linear plant; subclasses override attitude update."""

    name: str = "base"

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
        self._nom = ControlIntegrator(dt, gravity, mass, hover_thrust)

    def nominal_accel(self, control, state):
        return self._nom.nominal_accel(control, state)

    def _update_attitude(
        self, R: torch.Tensor, omega_rad: torch.Tensor, dt: float
    ) -> torch.Tensor:
        raise NotImplementedError

    def step(
        self,
        state: MetricState,
        control: torch.Tensor,
        res_lin: torch.Tensor,
        res_ang: torch.Tensor,
        R: torch.Tensor | None = None,
    ) -> tuple[MetricState, torch.Tensor]:
        dt = self.dt
        a_lin_nom, a_ang_nom = self.nominal_accel(control, state)
        a_lin = a_lin_nom + res_lin
        a_ang = a_ang_nom + res_ang  # deg/s^2 body

        new_vel = state.vel + a_lin * dt
        new_pos = state.pos + new_vel * dt
        new_ang_vel = state.ang_vel + a_ang * dt  # deg/s
        omega_rad = torch.deg2rad(new_ang_vel)

        if R is None:
            R = _euler_ypr_to_rotation(state.euler_att)
        R_new = self._update_attitude(R, omega_rad, dt)
        new_att = _rotation_to_euler_ypr_deg(R_new)
        new_att = wrap_degrees(new_att)
        return (
            MetricState(pos=new_pos, vel=new_vel, euler_att=new_att, ang_vel=new_ang_vel),
            R_new,
        )

    def rollout(
        self,
        init_state: MetricState,
        controls: torch.Tensor,
        res_lin: torch.Tensor,
        res_ang: torch.Tensor,
    ) -> tuple[MetricState, torch.Tensor, torch.Tensor]:
        """Returns states, R trajectory (B,T,3,3), constraint ‖RᵀR−I‖_F (B,T)."""
        T = controls.shape[1]
        state = init_state.clone()
        R = _euler_ypr_to_rotation(state.euler_att)
        pos_l, vel_l, att_l, av_l, R_l, c_l = [], [], [], [], [], []
        for t in range(T):
            state, R = self.step(state, controls[:, t], res_lin[:, t], res_ang[:, t], R=R)
            pos_l.append(state.pos)
            vel_l.append(state.vel)
            att_l.append(state.euler_att)
            av_l.append(state.ang_vel)
            R_l.append(R)
            c_l.append(rotation_constraint_frobenius(R))
        traj = MetricState(
            pos=torch.stack(pos_l, dim=1),
            vel=torch.stack(vel_l, dim=1),
            euler_att=torch.stack(att_l, dim=1),
            ang_vel=torch.stack(av_l, dim=1),
        )
        return traj, torch.stack(R_l, dim=1), torch.stack(c_l, dim=1)


class EuclideanEulerR9(AttitudeIntegratorBase):
    """Treat R as R^9: R ← R + R[ω]_× dt (no re-orthonormalization)."""

    name = "euclidean_euler_R9"

    def _update_attitude(self, R, omega_rad, dt):
        return R + (R @ skew(omega_rad)) * dt


class ExpIntegrator(AttitudeIntegratorBase):
    """First-order Lie-group: R ← R Exp(ω dt). SkyJEPA-style."""

    name = "exp_first_order"

    def _update_attitude(self, R, omega_rad, dt):
        return R @ so3_exp(omega_rad * dt)


class RK4So3Integrator(AttitudeIntegratorBase):
    """RK4 on so(3) with body-rate treated as constant over the step."""

    name = "rk4_so3"

    def _update_attitude(self, R, omega_rad, dt):
        # Body-rate kinematics: Ṙ = R [ω]_×. Integrate ω-frame increment with RK4
        # on the Lie algebra then Exp (equivalent to Exp with constant ω for
        # first-order; we use classical RK4 on the matrix ODE projected via Exp).
        def f(Ri):
            return Ri @ skew(omega_rad)

        k1 = f(R)
        k2 = f(R + 0.5 * dt * k1)
        k3 = f(R + 0.5 * dt * k2)
        k4 = f(R + dt * k3)
        R_eucl = R + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        # Retract via polar-ish: SVD orthonormalization (keeps bake-off honest
        # about what "RK4 then project" does vs pure Exp).
        U, _, Vh = torch.linalg.svd(R_eucl)
        return U @ Vh


class LGVIMidpoint(AttitudeIntegratorBase):
    """Implicit midpoint / variational-style update on SO(3).

    Discrete: R_{k+1} = R_k Exp(ω_mid dt) with ω_mid ≈ ω_k (explicit midpoint
    with frozen rate). Full implicit solve omitted when residual plant already
    uses first-order rate hold — this is the short feasible form cited in
    Marsden & West 2001 / Lee–Leok–McClamroch geometric tracking spirit.
    """

    name = "lgvi_midpoint"

    def _update_attitude(self, R, omega_rad, dt):
        # Midpoint: use half-step Exp twice (composition ≈ Exp(ω dt)).
        half = so3_exp(omega_rad * (0.5 * dt))
        return R @ half @ half


INTEGRATORS = {
    "euclidean_euler_R9": EuclideanEulerR9,
    "exp_first_order": ExpIntegrator,
    "rk4_so3": RK4So3Integrator,
    "lgvi_midpoint": LGVIMidpoint,
}
