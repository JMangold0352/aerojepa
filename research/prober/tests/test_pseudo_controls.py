"""Tests for Wilds gap-closure helpers (body→world + eval control priors)."""

from __future__ import annotations

import torch

from aerojepa_research.prober.pseudo_controls import (
    body_vel_to_world,
    make_eval_controls,
    metric_stack_body_vel_to_world,
)


def test_make_eval_controls_zeros_and_hover() -> None:
    z = make_eval_controls(5, mode="zeros")
    assert z.shape == (5, 4)
    assert torch.count_nonzero(z) == 0
    h = make_eval_controls(5, mode="hover", hover_thrust=0.39)
    assert torch.allclose(h[:, :3], torch.zeros(5, 3))
    assert torch.allclose(h[:, 3], torch.full((5,), 0.39))


def test_body_vel_to_world_yaw90() -> None:
    # Body +x with yaw=90° → world +y.
    vel = torch.tensor([[1.0, 0.0, 0.0]])
    att = torch.tensor([[90.0, 0.0, 0.0]])
    world = body_vel_to_world(vel, att)
    assert torch.allclose(world, torch.tensor([[0.0, 1.0, 0.0]]), atol=1e-5)


def test_metric_stack_rotates_only_velocity() -> None:
    stack = torch.zeros(1, 2, 12)
    stack[..., 3] = 1.0  # body vx
    stack[..., 6] = 90.0  # yaw
    out = metric_stack_body_vel_to_world(stack)
    assert torch.allclose(out[..., 0:3], stack[..., 0:3])
    assert torch.allclose(out[..., 3:6], torch.tensor([[[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]]), atol=1e-5)
    assert torch.allclose(out[..., 6:], stack[..., 6:])
