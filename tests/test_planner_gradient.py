from __future__ import annotations

import torch

from aerojepa.data.synthetic import sample_context
from aerojepa.models.jepa import AeroJEPA
from aerojepa.sim.planner import (
    LatentPlanner,
    MultiStepCostWeights,
    differentiable_plan_cost,
)
from aerojepa.utils.config import load_config


def _smoke_model():
    cfg = load_config("configs/smoke_test.yaml")
    model = AeroJEPA.from_config(cfg).to(torch.device("cpu")).eval()
    return model, cfg


def test_differentiable_cost_flows_gradients_to_actions() -> None:
    actions = (0.05 * torch.randn(4, 3, 6)).requires_grad_(True)
    goal = torch.tensor([0.3, 0.0, 0.0])
    cost, comps = differentiable_plan_cost(actions, goal=goal)
    assert cost.shape == (4,)
    # position, velocity, and attitude terms are all present and finite.
    for key in ("pos_terminal", "vel_terminal", "attitude", "attitude_rate"):
        assert key in comps
        assert torch.isfinite(comps[key]).all()
    cost.sum().backward()
    assert actions.grad is not None
    assert float(actions.grad.abs().sum()) > 0.0


def test_latent_term_only_added_when_weighted() -> None:
    actions = 0.05 * torch.randn(2, 3, 6)
    latents = torch.randn(2, 3, 8, 16)
    _, comps_off = differentiable_plan_cost(
        actions, pred_latents=latents, weights=MultiStepCostWeights(latent_smooth=0.0)
    )
    _, comps_on = differentiable_plan_cost(
        actions, pred_latents=latents, weights=MultiStepCostWeights(latent_smooth=0.1)
    )
    assert "latent_smooth" not in comps_off
    assert "latent_smooth" in comps_on


def test_gradient_planner_reduces_cost() -> None:
    model, cfg = _smoke_model()
    planner = LatentPlanner(
        model,
        torch.device("cpu"),
        planning="gradient",
        grad_steps=25,
        grad_lr=0.05,
    )
    ctx = sample_context(seed=3, context_frames=1, img_size=cfg["data"]["img_size"])[0]
    result = planner.plan(
        ctx, num_candidates=8, horizon=3, action_scale=0.10, goal=(0.3, 0.0, 0.0), seed=0
    )
    assert result.mode == "gradient"
    assert result.best_actions.shape == (3, 6)
    assert result.cost_history is not None and len(result.cost_history) == 25
    # Optimization must make progress.
    assert result.cost_history[-1] < result.cost_history[0]


def test_gradient_beats_shooting_on_differentiable_cost() -> None:
    model, cfg = _smoke_model()
    img = cfg["data"]["img_size"]
    ctx = sample_context(seed=5, context_frames=1, img_size=img)[0]
    goal = (0.3, 0.0, 0.0)
    gt = torch.tensor(goal)
    weights = MultiStepCostWeights(latent_smooth=0.0)

    grad = LatentPlanner(
        model, torch.device("cpu"), planning="gradient", grad_steps=30, grad_lr=0.05
    )
    g_res = grad.plan(ctx, num_candidates=8, horizon=3, action_scale=0.10, goal=goal, seed=0)

    shoot = LatentPlanner(model, torch.device("cpu"), planning="shooting")
    s_res = shoot.plan(ctx, num_candidates=64, horizon=3, action_scale=0.10, goal=goal, seed=0)

    g_cost, _ = differentiable_plan_cost(g_res.best_actions.unsqueeze(0), goal=gt, weights=weights)
    s_cost, _ = differentiable_plan_cost(s_res.best_actions.unsqueeze(0), goal=gt, weights=weights)
    assert float(g_cost) < float(s_cost)


def test_gradient_plan_reaches_goal() -> None:
    model, cfg = _smoke_model()
    ctx = sample_context(seed=7, context_frames=1, img_size=cfg["data"]["img_size"])[0]
    planner = LatentPlanner(
        model, torch.device("cpu"), planning="gradient", grad_steps=40, grad_lr=0.06
    )
    goal = torch.tensor([0.3, 0.0, 0.0])
    result = planner.plan(ctx, num_candidates=8, horizon=3, action_scale=0.10, goal=(0.3, 0.0, 0.0), seed=0)
    final_pos = torch.cumsum(result.best_actions[:, :3], dim=0)[-1]
    assert torch.norm(final_pos - goal) < 0.15
