from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import torch

from aerojepa.models.jepa import AeroJEPA

# Latent-space planning (research reference, not a flight controller).
#
# Sample candidate action sequences, roll them through the AeroJEPA predictor,
# score imagined futures, and keep the best plan. Works with action-conditioned
# checkpoints (latents depend on actions) and plain world models (falls back to
# kinematic costs). Closed-loop demos live in ``closed_loop.py`` with PyFlyt
# hooks in ``simulators.py``.

PredictorActionAblation = Literal["true", "zero", "shuffle"]


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
class MultiStepCostWeights:
    """Weights for the differentiable multi-step planning cost.

    The cost is a smooth, fully differentiable function of the *action plan*
    (and, optionally, the imagined latents), so gradients flow back to the
    actions and the plan can be optimized directly rather than only sampled.

    The kinematic state is read straight from the 6-DoF action deltas
    ``[dx, dy, d_alt, d_yaw, d_pitch, d_roll]``:

    * position  = cumulative translational deltas (relative to now)
    * velocity  = per-step translational delta
    * attitude  = cumulative rotational deltas
    * att. rate = per-step rotational delta
    """

    pos_terminal: float = 1.0  # be at the goal / origin at the end of the plan
    pos_running: float = 0.25  # ... and along the way (reach quickly, then hold)
    vel_terminal: float = 0.6  # arrive slow (kills overshoot / drift)
    vel_running: float = 0.05  # gentle damping throughout
    attitude: float = 0.3  # stay level (small accumulated tilt)
    attitude_rate: float = 0.1  # smooth attitude (no jerky rotation)
    effort: float = 0.02  # cheap control
    latent_smooth: float = 0.05  # couple the plan to the world model's latents


def differentiable_plan_cost(
    actions: torch.Tensor,
    *,
    goal: torch.Tensor | None = None,
    pred_latents: torch.Tensor | None = None,
    init_velocity: torch.Tensor | None = None,
    weights: MultiStepCostWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fully differentiable cost on position, velocity, and attitude.

    Parameters
    ----------
    actions : (N, H, 6) action plans (leaf that gradients flow back to).
    goal : (3,) desired net translational displacement, or ``None`` for hover
        (hold the current position).
    pred_latents : (N, H, S, D) imagined future latents. When provided together
        with ``weights.latent_smooth > 0``, a latent-smoothness term couples the
        plan to the world model - this is what makes it planning *in latent
        space* rather than pure kinematics.
    init_velocity : (3,) current translational velocity carried into the plan
        (a first-order momentum model). When given, the effective per-step
        velocity is ``planned_delta + init_velocity``, so the optimizer plans
        *braking* actions to arrest existing motion - essential for recovering
        from a disturbance rather than only chasing a position goal.

    Returns
    -------
    cost : (N,) per-candidate total cost.
    components : dict of per-candidate terms (for logging / tests).
    """
    w = weights or MultiStepCostWeights()
    trans = actions[..., :3]  # (N, H, 3) planned per-step translation
    rot = actions[..., 3:6]  # (N, H, 3) per-step rotation == angular-rate proxy
    if init_velocity is not None:
        v0 = init_velocity.to(actions.device, actions.dtype).view(1, 1, 3)
        velocity = trans + v0  # effective motion including carried momentum
    else:
        velocity = trans
    position = torch.cumsum(velocity, dim=1)  # (N, H, 3)
    attitude = torch.cumsum(rot, dim=1)  # (N, H, 3)

    if goal is not None:
        g = goal.to(actions.device, actions.dtype).view(1, 1, 3)
        pos_terminal = (position[:, -1:] - g).pow(2).sum(-1).mean(1)
        pos_running = (position - g).pow(2).sum(-1).mean(1)
    else:
        pos_terminal = position[:, -1].pow(2).sum(-1)
        pos_running = position.pow(2).sum(-1).mean(1)

    vel_terminal = velocity[:, -1].pow(2).sum(-1)
    vel_running = velocity.pow(2).sum(-1).mean(1)
    att_term = attitude.pow(2).sum(-1).mean(1)
    att_rate = rot.pow(2).sum(-1).mean(1)
    effort = actions.pow(2).mean(dim=(1, 2))

    cost = (
        w.pos_terminal * pos_terminal
        + w.pos_running * pos_running
        + w.vel_terminal * vel_terminal
        + w.vel_running * vel_running
        + w.attitude * att_term
        + w.attitude_rate * att_rate
        + w.effort * effort
    )
    components = {
        "pos_terminal": pos_terminal,
        "pos_running": pos_running,
        "vel_terminal": vel_terminal,
        "vel_running": vel_running,
        "attitude": att_term,
        "attitude_rate": att_rate,
        "effort": effort,
    }
    if pred_latents is not None and w.latent_smooth > 0:
        diffs = pred_latents[:, 1:] - pred_latents[:, :-1]
        latent_smooth = diffs.pow(2).mean(dim=(1, 2, 3))
        cost = cost + w.latent_smooth * latent_smooth
        components["latent_smooth"] = latent_smooth
    return cost, components


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
    mode: str = "shooting"  # "shooting" (random) or "gradient" (optimized)
    cost_history: list[float] | None = None  # per-iter best cost (gradient mode)

    @property
    def best_latents(self) -> torch.Tensor:
        """Imagined latents for the winning plan: ``(horizon, num_spatial, dim)``."""
        return self.pred_latents[self.best_index]


class LatentPlanner:
    """Random-shooting planner that imagines action sequences with the world model.

    Optional ``residual_head`` does **not** change imagination / scoring - it is
    only consulted by closed-loop execution helpers when mapping the winning
    AeroJEPA actions to PyFlyt controls.
    """

    def __init__(
        self,
        model: AeroJEPA,
        device: torch.device,
        cost_fn: LatentCost | str = hover_cost,
        action_dim: int = 6,
        residual_head: torch.nn.Module | None = None,
        planning: str = "shooting",
        grad_steps: int = 30,
        grad_lr: float = 0.05,
        grad_action_limit: float = 0.5,
        latent_refine_steps: int = 8,
        cost_weights: MultiStepCostWeights | None = None,
        predictor_action_ablation: PredictorActionAblation = "true",
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.cost_fn = COST_FUNCTIONS[cost_fn] if isinstance(cost_fn, str) else cost_fn
        self.action_dim = action_dim
        self.action_conditioned = model.predictor_is_action_conditioned()
        self.residual_head = residual_head.eval() if residual_head is not None else None
        if self.residual_head is not None:
            for p in self.residual_head.parameters():
                p.requires_grad = False
        if planning not in ("shooting", "gradient"):
            raise ValueError(f"planning must be 'shooting' or 'gradient', got {planning!r}")
        self.planning = planning
        self.grad_steps = grad_steps
        self.grad_lr = grad_lr
        self.grad_action_limit = grad_action_limit
        self.latent_refine_steps = latent_refine_steps
        self.cost_weights = cost_weights or MultiStepCostWeights()
        if predictor_action_ablation not in ("true", "zero", "shuffle"):
            raise ValueError(
                f"predictor_action_ablation must be true|zero|shuffle, "
                f"got {predictor_action_ablation!r}"
            )
        self.predictor_action_ablation: PredictorActionAblation = predictor_action_ablation
        self._ablation_perm: torch.Tensor | None = None

    def _target_indices(
        self, n: int, context_frames: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Context / target token indices for ``n`` candidates."""
        num_spatial = self.model.encoder.num_spatial
        num_temporal = self.model.encoder.num_temporal
        ctx = torch.arange(0, context_frames * num_spatial, device=self.device)
        tgt = torch.arange(
            context_frames * num_spatial, num_temporal * num_spatial, device=self.device
        )
        return ctx.unsqueeze(0).expand(n, -1), tgt.unsqueeze(0).expand(n, -1)

    def _encode_context(self, context_clip: torch.Tensor, context_frames: int) -> torch.Tensor:
        """Encode the context frames once (no grad). Returns ``(1, n_ctx, D)``."""
        num_temporal = self.model.encoder.num_temporal
        horizon = num_temporal - context_frames
        clip = context_clip.unsqueeze(0).to(self.device)
        pad = clip[:, -1:].expand(1, horizon, -1, -1, -1)
        full_clip = torch.cat([clip, pad], dim=1)
        ctx, _ = self._target_indices(1, context_frames)
        with torch.no_grad():
            return self.model.encoder(full_clip, ctx).detach()

    def _predict_latents(
        self,
        context_repr: torch.Tensor,
        context_frames: int,
        actions_full: torch.Tensor,
    ) -> torch.Tensor:
        """Predictor pass → ``(N, horizon, num_spatial, encoder_dim)``.

        Differentiable w.r.t. ``actions_full``; honours the current autograd
        mode so callers can wrap it in ``no_grad`` (shooting) or ``enable_grad``
        (gradient planning).
        """
        num_spatial = self.model.encoder.num_spatial
        num_temporal = self.model.encoder.num_temporal
        n = actions_full.shape[0]
        horizon = num_temporal - context_frames
        ctx, tgt = self._target_indices(n, context_frames)
        acts = actions_full.to(self.device) if self.action_conditioned else None
        if acts is not None and self.predictor_action_ablation == "zero":
            # Same contract as eval_action_counterfactual: zero the action tensor
            # the predictor sees. Planned actions still execute via the map.
            acts = torch.zeros_like(acts)
        elif acts is not None and self.predictor_action_ablation == "shuffle":
            # Permute across the candidate batch (fixed for one plan() call).
            if self._ablation_perm is None or int(self._ablation_perm.shape[0]) != n:
                self._ablation_perm = torch.randperm(n, device=acts.device)
            acts = acts[self._ablation_perm]
        out = self.model.predictor(context_repr, ctx, tgt, acts)
        if isinstance(out, tuple):  # looped predictor with exit gate
            out = out[0]
        return out.reshape(n, horizon, num_spatial, -1)

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
        n = actions_full.shape[0]
        context_repr = self._encode_context(context_clip, context_frames).expand(n, -1, -1)
        return self._predict_latents(context_repr, context_frames, actions_full)

    @staticmethod
    def _coherence(latents: torch.Tensor) -> float:
        """Mean cosine between consecutive imagined latent frames (temporal coherence)."""
        if latents.shape[0] < 2:
            return 1.0
        a = latents[:-1].flatten(1)
        b = latents[1:].flatten(1)
        return float(torch.cosine_similarity(a, b, dim=-1).mean().item())

    def _resolve_horizon(self, context_frames: int, horizon: int | None) -> int:
        num_temporal = self.model.encoder.num_temporal
        if horizon is None:
            horizon = num_temporal - context_frames
        if horizon <= 0 or context_frames + horizon > num_temporal:
            raise ValueError(
                f"horizon must satisfy 0 < context_frames + horizon <= {num_temporal}; "
                f"got context_frames={context_frames}, horizon={horizon}."
            )
        return horizon

    def _init_candidates(
        self,
        num_candidates: int,
        horizon: int,
        action_scale: float,
        goal: tuple[float, float, float] | torch.Tensor | None,
        seed: int | None,
    ) -> torch.Tensor:
        """Sample initial action plans ``(N, horizon, action_dim)``.

        When a goal displacement is given, bias half the candidates toward a
        constant per-step move that would reach it (plus noise). Critical so both
        random shooting and gradient refinement start near a homing trajectory.
        """
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        plan_actions = action_scale * torch.randn(
            num_candidates, horizon, self.action_dim, generator=gen
        )
        if goal is not None:
            goal_t = torch.as_tensor(goal, dtype=torch.float32).reshape(3)
            n_dir = max(1, num_candidates // 2)
            step = (goal_t / float(horizon)).clamp(-0.25, 0.25)
            directed = action_scale * torch.randn(
                n_dir, horizon, self.action_dim, generator=gen
            )
            directed[:, :, :3] = step.view(1, 1, 3) + 0.35 * action_scale * torch.randn(
                n_dir, horizon, 3, generator=gen
            )
            plan_actions[:n_dir] = directed
        return plan_actions

    def _finalize(
        self,
        plan_actions: torch.Tensor,
        costs: torch.Tensor,
        pred: torch.Tensor,
        context_frames: int,
        horizon: int,
        mode: str,
        cost_history: list[float] | None = None,
    ) -> PlanResult:
        best = int(torch.argmin(costs).item())
        positions = torch.cumsum(plan_actions[:, :, :3], dim=1)
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
            mode=mode,
            cost_history=cost_history,
        )

    def plan(
        self,
        context_clip: torch.Tensor,
        num_candidates: int = 64,
        horizon: int | None = None,
        action_scale: float = 0.05,
        goal: tuple[float, float, float] | torch.Tensor | None = None,
        seed: int | None = None,
        init_velocity: tuple[float, float, float] | torch.Tensor | None = None,
    ) -> PlanResult:
        """Plan an action sequence for a single context clip.

        Dispatches to random shooting (``planning='shooting'``) or gradient-based
        multi-step optimization (``planning='gradient'``). ``init_velocity`` is
        only used by the gradient planner (momentum-aware braking).
        """
        if self.planning == "gradient":
            self._ablation_perm = None
            return self.plan_gradient(
                context_clip,
                num_candidates=num_candidates,
                horizon=horizon,
                action_scale=action_scale,
                goal=goal,
                seed=seed,
                init_velocity=init_velocity,
            )
        self._ablation_perm = None
        return self.plan_shooting(
            context_clip,
            num_candidates=num_candidates,
            horizon=horizon,
            action_scale=action_scale,
            goal=goal,
            seed=seed,
        )

    @torch.no_grad()
    def plan_shooting(
        self,
        context_clip: torch.Tensor,
        num_candidates: int = 64,
        horizon: int | None = None,
        action_scale: float = 0.05,
        goal: tuple[float, float, float] | torch.Tensor | None = None,
        seed: int | None = None,
    ) -> PlanResult:
        """Random-shooting planner: sample ``num_candidates`` plans, keep the best."""
        context_frames = context_clip.shape[0]
        horizon = self._resolve_horizon(context_frames, horizon)
        num_temporal = self.model.encoder.num_temporal

        plan_actions = self._init_candidates(
            num_candidates, horizon, action_scale, goal, seed
        )
        actions_full = torch.zeros(num_candidates, num_temporal, self.action_dim)
        actions_full[:, context_frames:] = plan_actions

        positions = torch.cumsum(plan_actions[:, :, :3], dim=1)
        pred = self._imagine(context_clip, actions_full, context_frames).cpu()

        goal_t = None if goal is None else torch.as_tensor(goal, dtype=torch.float32)
        rollout = PlanRollout(
            actions=plan_actions, positions=positions, pred_latents=pred, goal=goal_t
        )
        costs = self.cost_fn(rollout).cpu()
        return self._finalize(
            plan_actions, costs, pred, context_frames, horizon, mode="shooting"
        )

    def plan_gradient(
        self,
        context_clip: torch.Tensor,
        num_candidates: int = 16,
        horizon: int | None = None,
        action_scale: float = 0.05,
        goal: tuple[float, float, float] | torch.Tensor | None = None,
        seed: int | None = None,
        init_velocity: tuple[float, float, float] | torch.Tensor | None = None,
        grad_steps: int | None = None,
        grad_lr: float | None = None,
        action_limit: float | None = None,
    ) -> PlanResult:
        """Multi-step planning by gradient descent on the action plan.

        Starts from a small population of goal-biased candidates and refines them
        with Adam against :func:`differentiable_plan_cost` (a smooth cost on
        position, velocity, and attitude, optionally coupled to the world model's
        imagined latents). Gradients flow through the frozen action-conditioned
        predictor back to the actions, so the plan is *optimized*, not just
        sampled. Keeps the horizon small (3-5 steps) by design.
        """
        context_frames = context_clip.shape[0]
        horizon = self._resolve_horizon(context_frames, horizon)
        num_temporal = self.model.encoder.num_temporal
        steps = self.grad_steps if grad_steps is None else grad_steps
        lr = self.grad_lr if grad_lr is None else grad_lr
        action_limit = (
            self.grad_action_limit if action_limit is None else action_limit
        )
        goal_t = (
            None if goal is None else torch.as_tensor(goal, dtype=torch.float32).reshape(3)
        )
        v0 = (
            None
            if init_velocity is None
            else torch.as_tensor(init_velocity, dtype=torch.float32).reshape(3).to(self.device)
        )

        init = self._init_candidates(num_candidates, horizon, action_scale, goal, seed)
        plan_actions = init.clone().to(self.device).requires_grad_(True)

        # Encode context once; freeze the world model so only the actions receive
        # gradients (the predictor stays a fixed, differentiable dynamics model).
        context_repr = self._encode_context(context_clip, context_frames).expand(
            num_candidates, -1, -1
        )
        # Latent coupling is expensive: run cheap kinematic steps first, then
        # refine the last ``latent_refine_steps`` with the world-model latents
        # when ``latent_smooth > 0``.
        want_latents = self.action_conditioned and self.cost_weights.latent_smooth > 0
        refine_from = max(0, steps - self.latent_refine_steps) if want_latents else steps
        saved = [(p, p.requires_grad) for p in self.model.parameters()]
        for p, _ in saved:
            p.requires_grad_(False)

        opt = torch.optim.Adam([plan_actions], lr=lr)
        cost_history: list[float] = []
        try:
            for it in range(steps):
                opt.zero_grad(set_to_none=True)
                pred = None
                use_latents = want_latents and it >= refine_from
                if use_latents:
                    actions_full = torch.zeros(
                        num_candidates, num_temporal, self.action_dim, device=self.device
                    )
                    actions_full[:, context_frames:] = plan_actions
                    with torch.enable_grad():
                        pred = self._predict_latents(
                            context_repr, context_frames, actions_full
                        )
                with torch.enable_grad():
                    costs, _ = differentiable_plan_cost(
                        plan_actions,
                        goal=goal_t,
                        pred_latents=pred,
                        init_velocity=v0,
                        weights=self.cost_weights,
                    )
                    loss = costs.sum()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    plan_actions.clamp_(-action_limit, action_limit)
                cost_history.append(float(costs.min().detach().cpu()))
        finally:
            for p, flag in saved:
                p.requires_grad_(flag)

        # Final scored pass (no grad) for the returned latents + costs.
        with torch.no_grad():
            final_actions = plan_actions.detach()
            actions_full = torch.zeros(
                num_candidates, num_temporal, self.action_dim, device=self.device
            )
            actions_full[:, context_frames:] = final_actions
            pred = self._predict_latents(context_repr, context_frames, actions_full)
            costs, _ = differentiable_plan_cost(
                final_actions,
                goal=goal_t,
                pred_latents=pred,
                init_velocity=v0,
                weights=self.cost_weights,
            )
        return self._finalize(
            final_actions.cpu(),
            costs.cpu(),
            pred.cpu(),
            context_frames,
            horizon,
            mode="gradient",
            cost_history=cost_history,
        )
