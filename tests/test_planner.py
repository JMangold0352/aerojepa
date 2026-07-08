from __future__ import annotations

import torch

from aerojepa.data.synthetic import integrate_actions, render_poses, sample_context
from aerojepa.models.jepa import AeroJEPA
from aerojepa.sim.planner import (
    LatentPlanner,
    PlanRollout,
    hover_cost,
    waypoint_cost,
)
from aerojepa.sim.rollout_demo import plan_and_render
from aerojepa.utils.config import load_config


def _smoke_model():
    cfg = load_config("configs/smoke_test.yaml")
    model = AeroJEPA.from_config(cfg).to(torch.device("cpu")).eval()
    return model, cfg


def test_action_driven_renderer_shapes() -> None:
    frames, end_pose, world = sample_context(seed=1, context_frames=3, img_size=32)
    assert frames.shape == (3, 3, 32, 32)
    assert len(end_pose) == 4
    actions = 0.05 * torch.randn(4, 6)
    poses = integrate_actions(end_pose, actions)
    rolled = render_poses(world, poses, img_size=32)
    assert rolled.shape == (4, 3, 32, 32)
    assert rolled.min() >= 0.0 and rolled.max() <= 1.0


def test_cost_functions_score_candidates() -> None:
    n, horizon = 5, 3
    actions = torch.randn(n, horizon, 6)
    positions = torch.cumsum(actions[:, :, :3], dim=1)
    latents = torch.randn(n, horizon, 16, 8)
    rollout = PlanRollout(actions=actions, positions=positions, pred_latents=latents, goal=torch.tensor([0.3, 0.0, 0.0]))
    assert hover_cost(rollout).shape == (n,)
    assert waypoint_cost(rollout).shape == (n,)


def test_planner_returns_valid_plan() -> None:
    model, _ = _smoke_model()
    planner = LatentPlanner(model, torch.device("cpu"), cost_fn="hover")
    result = planner.plan(
        sample_context(seed=3, context_frames=2, img_size=32)[0],
        num_candidates=32,
        action_scale=0.05,
        seed=0,
    )
    assert result.best_actions.shape == (result.horizon, 6)
    assert result.costs.shape == (32,)
    assert 0 <= result.best_index < 32
    # The chosen candidate must be the cheapest one.
    assert torch.isclose(result.costs[result.best_index], result.costs.min())
    assert -1.0 <= result.coherence <= 1.0001


def test_waypoint_plan_moves_toward_goal() -> None:
    model, _ = _smoke_model()
    planner = LatentPlanner(model, torch.device("cpu"), cost_fn="waypoint")
    ctx = sample_context(seed=5, context_frames=2, img_size=32)[0]
    result = planner.plan(ctx, num_candidates=128, action_scale=0.08, goal=(0.3, 0.0, 0.0), seed=1)
    final = result.positions[result.best_index, -1]
    mean_final_x = result.positions[:, -1, 0].mean()
    # The chosen plan should end nearer the +x goal than the average candidate.
    assert final[0] > mean_final_x


def test_plan_and_render_end_to_end(tmp_path) -> None:
    model, cfg = _smoke_model()
    out = plan_and_render(
        model,
        torch.device("cpu"),
        img_size=cfg["data"]["img_size"],
        in_chans=cfg["data"].get("in_chans", 3),
        seed=7,
        task="hover",
        num_candidates=16,
        out_gif=tmp_path / "plan.gif",
        out_fig=tmp_path / "plan.png",
    )
    assert out.gif_path is not None and out.gif_path.exists()
    assert (tmp_path / "plan.png").exists()
    assert out.planned_frames.shape[0] == out.result.horizon
