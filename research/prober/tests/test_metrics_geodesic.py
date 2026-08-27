"""Tests for geodesic attitude metric (docs/CORRECTNESS.md V2)."""

from __future__ import annotations

import torch

from aerojepa_research.prober.metrics import compute_metrics, geodesic_attitude_error_deg


def test_geodesic_identity_is_zero():
    att = torch.tensor([[10.0, -5.0, 20.0]])
    err = geodesic_attitude_error_deg(att, att)
    assert float(err) < 0.05  # numerical noise from YPR ↔ R roundtrip


def test_geodesic_nonzero_for_yaw_offset():
    gt = torch.zeros(1, 3)
    pred = torch.tensor([[30.0, 0.0, 0.0]])
    err = float(geodesic_attitude_error_deg(pred, gt))
    assert 25.0 < err < 35.0


def test_compute_metrics_reports_geodesic_field():
    B, T = 4, 3
    gt = torch.zeros(B, T, 12)
    pred = gt.clone()
    pred[..., 6] = 5.0  # yaw offset
    m = compute_metrics(pred, gt)
    assert m.attitude_rmse_geodesic_deg > 0
    assert len(m.per_horizon_att_geodesic_rmse) == T
