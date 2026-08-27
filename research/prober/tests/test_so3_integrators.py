"""SO(3) integrator constraint tests."""

from __future__ import annotations

import torch

from aerojepa_research.prober.integrator import _euler_ypr_to_rotation
from aerojepa_research.prober.so3_integrators import (
    EuclideanEulerR9,
    ExpIntegrator,
    rotation_constraint_frobenius,
    so3_exp,
)


def test_exp_preserves_so3():
    R = so3_exp(torch.tensor([0.2, -0.1, 0.4]))
    err = float(rotation_constraint_frobenius(R))
    assert err < 1e-5


def test_euclidean_drifts_faster_than_exp():
    dt = 0.025
    steps = 200
    omega = torch.deg2rad(torch.tensor([120.0, -80.0, 40.0]))
    R0 = _euler_ypr_to_rotation(torch.tensor([[10.0, -5.0, 15.0]]))
    eu = EuclideanEulerR9(dt=dt)
    ex = ExpIntegrator(dt=dt)
    Re, Rx = R0.clone(), R0.clone()
    for _ in range(steps):
        Re = eu._update_attitude(Re, omega, dt)
        Rx = ex._update_attitude(Rx, omega, dt)
    assert float(rotation_constraint_frobenius(Re)) > 0.05
    assert float(rotation_constraint_frobenius(Rx)) < 1e-4
