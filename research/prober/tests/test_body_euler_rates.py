"""Unit tests for body-rate → Euler-rate map."""

from __future__ import annotations

import torch

from aerojepa_research.prober.integrator import body_rates_to_euler_rates_deg


def test_pure_yaw_rate_level():
    # Level craft: yaw rate ≈ body r
    att = torch.zeros(1, 3)
    omega = torch.tensor([[0.0, 0.0, 90.0]])  # deg/s about body z
    rates = body_rates_to_euler_rates_deg(att, omega)
    assert abs(float(rates[0, 0]) - 90.0) < 1e-3  # yaw_dot
    assert abs(float(rates[0, 1])) < 1e-3
    assert abs(float(rates[0, 2])) < 1e-3


def test_pure_roll_rate_level():
    att = torch.zeros(1, 3)
    omega = torch.tensor([[45.0, 0.0, 0.0]])  # p
    rates = body_rates_to_euler_rates_deg(att, omega)
    assert abs(float(rates[0, 2]) - 45.0) < 1e-3  # roll_dot
    assert abs(float(rates[0, 0])) < 1e-3
    assert abs(float(rates[0, 1])) < 1e-3
