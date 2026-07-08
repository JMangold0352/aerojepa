from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from aerojepa.data.synthetic import integrate_actions, render_poses, sample_context
from aerojepa.sim.planner import COST_FUNCTIONS, LatentPlanner, PlanResult

# One reusable entry point that runs the whole latent-planning story end to end:
#   1. render a short observed context clip from the synthetic camera,
#   2. imagine many action plans with the world model and pick the best,
#   3. execute the winning plan in the (pixel-space) simulator to see what it does,
#   4. optionally save an annotated GIF and a candidate-trajectory figure.
# Both scripts/run_planner_demo.py and the Gradio tab call this, so behavior is
# identical everywhere.


@dataclass
class PlanDemoOutput:
    result: PlanResult
    context_frames: torch.Tensor  # (context_frames, C, H, W)
    planned_frames: torch.Tensor  # (horizon, C, H, W) executed plan, ground truth
    gif_path: Path | None
    figure: object | None  # matplotlib Figure or None


def plan_and_render(
    model,
    device: torch.device,
    *,
    img_size: int = 64,
    in_chans: int = 3,
    num_obstacles: int = 5,
    seed: int = 7,
    task: str = "hover",
    context_frames: int | None = None,
    horizon: int | None = None,
    num_candidates: int = 64,
    action_scale: float = 0.06,
    goal: tuple[float, float, float] | None = None,
    out_gif: str | Path | None = None,
    out_fig: str | Path | None = None,
    make_figure: bool = True,
) -> PlanDemoOutput:
    """Plan with the world model and render the observed context + executed plan.

    ``task`` selects the cost function: ``"hover"`` (hold position), ``"waypoint"``
    (reach ``goal``), or ``"smoothness"`` (coherent imagined scene). For the
    waypoint task pass ``goal=(dx, dy, dz)``.
    """
    if task not in COST_FUNCTIONS:
        raise ValueError(f"task must be one of {sorted(COST_FUNCTIONS)}; got {task!r}")
    if task == "waypoint" and goal is None:
        goal = (0.25, 0.0, 0.0)  # a sensible default: fly right by 0.25 world units

    num_temporal = model.encoder.num_temporal
    if context_frames is None:
        context_frames = max(1, num_temporal // 2)
    context_frames = max(1, min(context_frames, num_temporal - 1))

    frames, end_pose, world = sample_context(
        seed, context_frames, img_size=img_size, in_chans=in_chans, num_obstacles=num_obstacles
    )

    planner = LatentPlanner(model, device, cost_fn=task)
    result = planner.plan(
        frames,
        num_candidates=num_candidates,
        horizon=horizon,
        action_scale=action_scale,
        goal=goal,
        seed=seed,
    )

    # Execute the winning plan in the pixel-space simulator to see what happens.
    planned_poses = integrate_actions(end_pose, result.best_actions)
    planned_frames = render_poses(world, planned_poses, img_size=img_size, in_chans=in_chans)

    gif_path = None
    if out_gif is not None:
        from aerojepa.viz.planner_viz import render_plan_gif

        gif_path = render_plan_gif(
            frames,
            planned_frames,
            out_gif,
            coherence=result.coherence,
            cost=float(result.costs[result.best_index]),
        )

    figure = None
    if make_figure or out_fig is not None:
        from aerojepa.viz.planner_viz import plan_trajectory_figure

        figure = plan_trajectory_figure(result, out_fig)

    return PlanDemoOutput(
        result=result,
        context_frames=frames,
        planned_frames=planned_frames,
        gif_path=gif_path,
        figure=figure,
    )
