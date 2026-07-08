from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from aerojepa.models.jepa import AeroJEPA

# Predictive planning in latent space -- a deliberately compact, readable
# reference implementation.
#
# The promise of a world model is that you can plan *without touching the real
# world*: imagine several candidate action sequences, roll each one forward in
# latent space, score the imagined futures, and execute the best one. This module
# implements the simplest version of that loop -- random shooting (sample many
# action sequences, keep the best) -- on top of an AeroJEPA predictor.
#
# It works with two kinds of checkpoint:
#   * action-conditioned world model -- the predicted latents genuinely depend on
#     the candidate actions, so cost functions can score the *imagined scene*.
#   * plain world model (no action input) -- the model still imagines a coherent
#     future, but it cannot tell candidates apart; planning then falls back to a
#     kinematic cost over the action/position trajectory (still useful for hover
#     and waypoint tasks). The planner reports which mode it is in.
#
# This is a research reference, not a flight controller. The cost function is
# pluggable; full closed-loop control lives behind the simulator hooks in
# ``simulators.py`` (see docs/ROADMAP.md, Phase 4).


@dataclass
class PlanRollout:
    """Everything a cost function might need to score one batch of candidates."""

    actions: torch.Tensor  # (N, horizon, action_dim) candidate action plans
    positions: torch.Tensor  # (N, horizon, 3) integrated xyz trajectory
    pred_latents: torch.Tensor  # (N, horizon, num_spatial, dim) imagined future
    goal: torch.Tensor | None  # (3,) desired net displacement, or None


LatentCost = Callable[[PlanRollout], torch.Tensor]


def smoothness_cost(rollout: PlanRollout) -> torch.Tensor:
    """Prefer futures whose *imagined latents* evolve smoothly in time.

    Only discriminative for action-conditioned models (otherwise every candidate
    shares the same imagined future). A good generic "don't do anything jarring"
    objective and the closest thing to scoring the scene itself.
    """
    lat = rollout.pred_latents
    diffs = lat[:, 1:] - lat[:, :-1]
    return diffs.pow(2).mean(dim=(1, 2, 3))


def hover_cost(rollout: PlanRollout) -> torch.Tensor:
    """Stay put: penalize drift away from the start and any wasted effort.

    The canonical quadrotor stability task -- hold position. Works for any
    checkpoint because it scores the kinematic trajectory, not the latents.
    """
    drift = rollout.positions.pow(2).sum(dim=-1).mean(dim=1)
    effort = rollout.actions.pow(2).mean(dim=(1, 2))
    return drift + 0.1 * effort


def waypoint_cost(rollout: PlanRollout) -> torch.Tensor:
    """Reach a goal: penalize the final position's distance to ``goal`` plus effort."""
    if rollout.goal is None:
        raise ValueError("waypoint_cost needs a goal; pass goal=(dx, dy, dz) to plan().")
    goal = rollout.goal.to(rollout.positions.device)
    final = rollout.positions[:, -1, :]
    return (final - goal).pow(2).sum(dim=-1) + 0.05 * rollout.actions.pow(2).mean(dim=(1, 2))


COST_FUNCTIONS: dict[str, LatentCost] = {
    "hover": hover_cost,
    "waypoint": waypoint_cost,
    "smoothness": smoothness_cost,
}


@dataclass
class PlanResult:
    """The chosen plan plus everything needed to visualize or execute it."""

    best_actions: torch.Tensor  # (horizon, action_dim) plan to execute
    best_index: int
    costs: torch.Tensor  # (N,) cost of every candidate
    positions: torch.Tensor  # (N, horizon, 3) all candidate trajectories
    pred_latents: torch.Tensor  # (N, horizon, num_spatial, dim)
    coherence: float  # mean cosine between consecutive imagined latent frames
    action_conditioned: bool  # True if the model scored candidates via its latents
    context_frames: int
    horizon: int


class LatentPlanner:
    """Random-shooting planner that imagines action sequences with the world model."""

    def __init__(
        self,
        model: AeroJEPA,
        device: torch.device,
        cost_fn: LatentCost | str = hover_cost,
        action_dim: int = 6,
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.cost_fn = COST_FUNCTIONS[cost_fn] if isinstance(cost_fn, str) else cost_fn
        self.action_dim = action_dim
        self.action_conditioned = model.predictor_is_action_conditioned()

    @torch.no_grad()
    def _imagine(
        self,
        context_clip: torch.Tensor,
        actions_full: torch.Tensor,
        context_frames: int,
    ) -> torch.Tensor:
        """Predict future-frame latents for a batch of candidate action plans.

        Returns ``(N, horizon, num_spatial, encoder_dim)``.
        """
        num_spatial = self.model.encoder.num_spatial
        num_temporal = self.model.encoder.num_temporal
        n = actions_full.shape[0]
        horizon = num_temporal - context_frames

        clip = context_clip.unsqueeze(0).expand(n, -1, -1, -1, -1).to(self.device)
        # Pad to full clip length with the last observed frame; only the context
        # tokens are read by the encoder, so this padding never leaks into targets.
        pad = clip[:, -1:].expand(n, horizon, -1, -1, -1)
        full_clip = torch.cat([clip, pad], dim=1)

        ctx = torch.arange(0, context_frames * num_spatial, device=self.device)
        tgt = torch.arange(context_frames * num_spatial, num_temporal * num_spatial, device=self.device)
        ctx = ctx.unsqueeze(0).expand(n, -1)
        tgt = tgt.unsqueeze(0).expand(n, -1)

        acts = actions_full.to(self.device) if self.action_conditioned else None
        out = self.model(full_clip, ctx, tgt, actions=acts)
        return out["pred_repr"].reshape(n, horizon, num_spatial, -1)

    @staticmethod
    def _coherence(latents: torch.Tensor) -> float:
        """Mean cosine between consecutive imagined latent frames (temporal coherence)."""
        if latents.shape[0] < 2:
            return 1.0
        a = latents[:-1].flatten(1)
        b = latents[1:].flatten(1)
        return float(torch.cosine_similarity(a, b, dim=-1).mean().item())

    @torch.no_grad()
    def plan(
        self,
        context_clip: torch.Tensor,
        num_candidates: int = 64,
        horizon: int | None = None,
        action_scale: float = 0.05,
        goal: tuple[float, float, float] | torch.Tensor | None = None,
        seed: int | None = None,
    ) -> PlanResult:
        """Plan an action sequence for a single context clip.

        ``context_clip`` is ``(context_frames, C, H, W)``. We sample
        ``num_candidates`` random 6-DoF action plans of length ``horizon``,
        imagine each one with the world model, score them with ``cost_fn``, and
        return the lowest-cost plan together with the imagined futures.
        """
        num_temporal = self.model.encoder.num_temporal
        context_frames = context_clip.shape[0]
        if horizon is None:
            horizon = num_temporal - context_frames
        if horizon <= 0 or context_frames + horizon > num_temporal:
            raise ValueError(
                f"horizon must satisfy 0 < context_frames + horizon <= {num_temporal}; "
                f"got context_frames={context_frames}, horizon={horizon}."
            )

        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        plan_actions = action_scale * torch.randn(
            num_candidates, horizon, self.action_dim, generator=gen
        )
        # Full-length action tensor: context frames get zero motion (already
        # observed), future frames get the candidate plan.
        actions_full = torch.zeros(num_candidates, num_temporal, self.action_dim)
        actions_full[:, context_frames:] = plan_actions

        positions = torch.cumsum(plan_actions[:, :, :3], dim=1)
        pred = self._imagine(context_clip, actions_full, context_frames).cpu()

        goal_t = None if goal is None else torch.as_tensor(goal, dtype=torch.float32)
        rollout = PlanRollout(actions=plan_actions, positions=positions, pred_latents=pred, goal=goal_t)
        costs = self.cost_fn(rollout).cpu()
        best = int(torch.argmin(costs).item())

        return PlanResult(
            best_actions=plan_actions[best],
            best_index=best,
            costs=costs,
            positions=positions,
            pred_latents=pred,
            coherence=self._coherence(pred[best]),
            action_conditioned=self.action_conditioned,
            context_frames=context_frames,
            horizon=horizon,
        )
