from __future__ import annotations

import torch

from aerojepa.sim.action_residual import (
    ActionResidualHead,
    aerojepa_to_pyflyt_torch,
    apply_residual_control,
    residual_loss,
)


def test_residual_head_shapes_and_param_budget() -> None:
    head = ActionResidualHead(latent_dim=192, hidden_dim=16)
    assert head.num_params() < 5000
    b, t = 2, 4
    lat = torch.randn(b, t, 192)
    aero = torch.randn(b, t, 6) * 0.05
    heur = aerojepa_to_pyflyt_torch(aero)
    delta = head(lat, heur, aero)
    assert delta.shape == (b, t, 4)


def test_apply_residual_stays_in_bounds() -> None:
    head = ActionResidualHead(latent_dim=64, hidden_dim=8)
    # Overwrite final layer to large values to stress clamping.
    with torch.no_grad():
        head.mlp[-1].weight.fill_(1.0)
        head.mlp[-1].bias.fill_(2.0)
    aero = torch.zeros(6)
    lat = torch.zeros(64)
    ctrl = apply_residual_control(aero, lat, head)
    assert ctrl.shape == (4,)
    assert float(ctrl[3].detach()) <= 0.8 + 1e-5
    assert abs(float(ctrl[0].detach())) <= 3.15


def test_residual_loss_beats_or_matches_init_heur_scale() -> None:
    head = ActionResidualHead(latent_dim=32, hidden_dim=8)
    lat = torch.randn(4, 3, 32)
    aero = 0.05 * torch.randn(4, 3, 6)
    gt = aerojepa_to_pyflyt_torch(aero)  # perfect heuristic target
    loss, stats = residual_loss(head, lat, aero, gt, residual_l2=0.0)
    assert loss.ndim == 0
    assert stats["heur_mse"] < 1e-6
    # With zero residual init (near-zero last layer), mse ≈ heur_mse.
    assert stats["mse"] < 0.05
