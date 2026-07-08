"""Unit tests for the kinematic integrator.

These verify the core invariants: zero-residual + nominal-action reproduces the
nominal trajectory, wrapping keeps attitude in (-180, 180], and the whole thing
is differentiable (gradients flow to residuals).
"""

from __future__ import annotations

import pytest
import torch

from aerojepa_research.prober.integrator import (
    KinematicIntegrator,
    MetricState,
    wrap_degrees,
)


def test_wrap_degrees_basic():
    # The formula (x + 180) % 360 - 180 maps to [-180, 180): 180 -> -180.
    # This matches the original telemetry.wrap_degrees helper exactly.
    x = torch.tensor([0.0, 180.0, 181.0, 359.0, -181.0, 360.0])
    out = wrap_degrees(x)
    assert torch.all(out >= -180.0)
    assert torch.all(out < 180.0)
    # 181 -> -179
    assert abs(out[2].item() - (-179.0)) < 1e-5
    # 359 -> -1
    assert abs(out[3].item() - (-1.0)) < 1e-5


def test_step_zero_residual_matches_nominal():
    """With zero residual, the integrator should track the action's velocity."""
    dt = 1.0 / 15.0
    integ = KinematicIntegrator(dt=dt, gravity=0.0)  # disable gravity for clarity
    B = 2
    state = MetricState.zeros(B)
    # Action: body velocity (1, 0, 0), attitude delta (0, 0, 0)
    action = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]] * B)
    res_lin = torch.zeros(B, 3)
    res_ang = torch.zeros(B, 3)

    next_state = integ.step(state, action, res_lin, res_ang)
    # Nominal accel = (action_vel - vel)/dt = (1 - 0)/dt. new_vel = 0 + a*dt = 1.
    assert torch.allclose(next_state.vel, action[..., :3], atol=1e-5)
    # new_pos = pos + new_vel*dt = 0 + 1*dt = dt
    assert torch.allclose(next_state.pos, torch.full((B, 3), dt) * torch.tensor([1.0, 0.0, 0.0]), atol=1e-5)


def test_step_gravity_pulls_down_on_zero_action():
    """With zero action and zero residual, gravity should pull z-velocity down."""
    dt = 0.1
    g = -9.81
    integ = KinematicIntegrator(dt=dt, gravity=g)
    state = MetricState.zeros(1)
    action = torch.zeros(1, 6)
    res = torch.zeros(1, 3)

    next_state = integ.step(state, action, res, res)
    # a_lin_nom z = (0 - 0)/dt + g = g. new_vel_z = 0 + g*dt.
    assert abs(next_state.vel[0, 2].item() - g * dt) < 1e-5
    # pos z = 0 + new_vel*dt = g*dt^2
    assert abs(next_state.pos[0, 2].item() - g * dt * dt) < 1e-5


def test_rollout_horizon_and_shapes():
    integ = KinematicIntegrator(dt=0.1, gravity=0.0)
    B, T = 3, 5
    init = MetricState.zeros(B)
    actions = torch.randn(B, T, 6)
    res_lin = torch.randn(B, T, 3)
    res_ang = torch.randn(B, T, 3)

    traj = integ.rollout(init, actions, res_lin, res_ang)
    assert traj.pos.shape == (B, T, 3)
    assert traj.vel.shape == (B, T, 3)
    assert traj.euler_att.shape == (B, T, 3)
    assert traj.ang_vel.shape == (B, T, 3)


def test_rollout_attitude_stays_wrapped():
    """Long rollouts with large angular velocities must keep attitude in range."""
    integ = KinematicIntegrator(dt=0.1, gravity=0.0)
    B, T = 1, 50
    init = MetricState.zeros(B)
    # Huge attitude deltas to force wrapping.
    actions = torch.zeros(B, T, 6)
    actions[..., 3] = 200.0  # d_yaw = 200 deg/frame
    res_lin = torch.zeros(B, T, 3)
    res_ang = torch.zeros(B, T, 3)

    traj = integ.rollout(init, actions, res_lin, res_ang)
    # [-180, 180) per the wrap formula.
    assert torch.all(traj.euler_att >= -180.0)
    assert torch.all(traj.euler_att < 180.0)


def test_differentiable_to_residuals():
    """Gradients must flow back to both residuals (the prober's outputs).

    A position-only loss won't reach res_ang (angular dynamics don't affect
    position in the nominal model), so we use a loss over the full state.
    """
    integ = KinematicIntegrator(dt=0.1, gravity=0.0)
    B, T = 2, 4
    init = MetricState.zeros(B)
    actions = torch.randn(B, T, 6)
    res_lin = torch.randn(B, T, 3, requires_grad=True)
    res_ang = torch.randn(B, T, 3, requires_grad=True)

    traj = integ.rollout(init, actions, res_lin, res_ang)
    # Loss over all state components so both residual tensors get gradients.
    loss = traj.stack().pow(2).sum()
    loss.backward()
    assert res_lin.grad is not None
    assert res_ang.grad is not None
    assert torch.any(res_lin.grad != 0)
    assert torch.any(res_ang.grad != 0)


def test_from_stack_roundtrip():
    B = 3
    state = MetricState(
        pos=torch.randn(B, 3),
        vel=torch.randn(B, 3),
        euler_att=torch.randn(B, 3) * 30.0,
        ang_vel=torch.randn(B, 3),
    )
    stacked = state.stack()
    assert stacked.shape == (B, 12)
    recovered = MetricState.from_stack(stacked)
    assert torch.allclose(recovered.pos, state.pos)
    assert torch.allclose(recovered.vel, state.vel)
    assert torch.allclose(recovered.euler_att, state.euler_att)
    assert torch.allclose(recovered.ang_vel, state.ang_vel)
