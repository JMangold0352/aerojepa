"""Smoke: PlanningForward matches LatentPlanner imagine shapes (no residual)."""

from __future__ import annotations

import torch

from aerojepa.export.planning_forward import build_planning_forward, planning_shapes
from aerojepa.models.jepa import AeroJEPA
from aerojepa.sim.planner import LatentPlanner
from aerojepa.utils.config import load_config


def _smoke_model():
    cfg = load_config("configs/smoke_test.yaml")
    model = AeroJEPA.from_config(cfg).to(torch.device("cpu")).eval()
    return model, cfg


def test_planning_forward_matches_imagine() -> None:
    model, cfg = _smoke_model()
    img = int(cfg["data"]["img_size"])
    pf = build_planning_forward(model)
    shapes = planning_shapes(model, img_size=img)
    assert shapes["horizon"] == pf.horizon
    assert shapes["param_count"] == pf.param_count()
    assert shapes["param_count"] < 6_000_000

    b = 2
    ctx_one = torch.rand(pf.context_frames, 3, img, img)
    context = ctx_one.unsqueeze(0).expand(b, -1, -1, -1, -1).contiguous()
    actions = torch.zeros(b, pf.num_temporal, 6)
    with torch.no_grad():
        out = pf(context, actions)
    assert out.shape == (b, pf.horizon, pf.num_spatial, pf.encoder_dim)

    # Planner encodes once and expands; same clip → same latents as batched forward.
    planner = LatentPlanner(model, torch.device("cpu"), cost_fn="hover")
    with torch.no_grad():
        ref = planner._imagine(ctx_one, actions, pf.context_frames)
    assert ref.shape == out.shape
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_planning_forward_torchscript_trace() -> None:
    model, cfg = _smoke_model()
    img = int(cfg["data"]["img_size"])
    pf = build_planning_forward(model)
    context = torch.rand(1, pf.context_frames, 3, img, img)
    actions = torch.zeros(1, pf.num_temporal, 6)
    with torch.no_grad():
        traced = torch.jit.trace(pf, (context, actions), strict=False)
        a = pf(context, actions)
        b = traced(context, actions)
    assert a.shape == b.shape
    assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
