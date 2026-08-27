"""Closed-loop PyFlyt evaluation of the AeroJEPA latent planner.

Bridges the research gap between :class:`~aerojepa.sim.planner.LatentPlanner`
(plans in AeroJEPA's 6-DoF action space from egocentric video) and a real
physics simulator (PyFlyt QuadX, which expects angular-rate + thrust commands).

This is a research demo loop - not a flight controller. The action map is a
deliberately simple heuristic so the full stack (camera → world model → plan →
physics) is runnable end-to-end and comparable against inert / random / seek
baselines.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from aerojepa.models.jepa import AeroJEPA
from aerojepa.sim.action_residual import ActionResidualHead, map_aero_with_optional_residual
from aerojepa.sim.planner import COST_FUNCTIONS, LatentPlanner, MultiStepCostWeights
from aerojepa.sim.simulators import make_pyflyt_env

PolicyName = Literal["planner", "hover", "random", "inert", "seek"]

# Empirically near hover for QuadX-Hover-v4 with default mass / thrust limits.
DEFAULT_HOVER_THRUST = 0.39
DEFAULT_ENV_ID = "PyFlyt/QuadX-Hover-v4"
DEFAULT_WAYPOINT_GOAL = (0.6, 0.0, 0.0)
DEFAULT_REACH_THRESHOLD = 0.25
DEFAULT_RECOVER_XY_THRESHOLD = 0.40
DEFAULT_DISTURB_AT = 30
DEFAULT_DISTURB_STEPS = 18
# Empirically +vq moves +x on QuadX-Hover-v4. Sized so a pure hover
# hold after the kick stays displaced (~0.5-1 m), while seek/planner can home.
DEFAULT_DISTURB_ACTION = (0.0, 0.75, 0.0, DEFAULT_HOVER_THRUST)
DEFAULT_MIN_KICK_XY = 0.35
DEFAULT_DAMP_STEPS = 15
# Adaptive braking after the kick: keep damping while lateral speed is above
# this (m/s), up to DEFAULT_DAMP_MAX_STEPS total. Passive hover lets the craft
# coast 1.5+ m and come home hot enough to tip over.
DEFAULT_BRAKE_VEL = 0.6
DEFAULT_DAMP_MAX_STEPS = 60

# Stress-test defaults (moderate wind first; keep heuristic map unchanged).
DEFAULT_WIND_MPS = 2.0  # m/s constant lateral gust - noticeable but hoverable
DEFAULT_WIND_ONSET = 40  # settle steps before wind engages
DEFAULT_WIND_DRIFT_FAIL = 1.5  # meters; beyond this under wind = excessive_drift
DEFAULT_AGGRESSIVE_LEG1 = (0.5, 0.0, 0.0)  # first leg of the L-turn (survivable)
DEFAULT_AGGRESSIVE_LEG2 = (0.5, 0.5, 0.0)  # 90° corner; shorter than the hard 0.8 m course
DEFAULT_AGGRESSIVE_LEG1_HARD = (0.8, 0.0, 0.0)
DEFAULT_AGGRESSIVE_LEG2_HARD = (0.8, 0.8, 0.0)
# Tip-in at the corner rarely tags the tight hover threshold.
DEFAULT_AGGRESSIVE_REACH = 0.35
# Early-turn look-ahead: start aiming at leg2 before tagging leg1.
DEFAULT_CORNER_LOOKAHEAD = 0.55
# Light seek PD blend on the L-turn (leg2 gets more; keeps tip-over in check).
DEFAULT_AGGRESSIVE_SEEK_BLEND_LEG1 = 0.20
DEFAULT_AGGRESSIVE_SEEK_BLEND_LEG2 = 0.45
DEFAULT_ALTITUDE_COLLAPSE = 0.35  # meters; below this while still flying = collapse

STRESS_TASKS = frozenset({"wind_gust", "aggressive_turn"})
ALL_TASKS = frozenset(set(COST_FUNCTIONS) | {"recover"} | STRESS_TASKS)


def aerojepa_to_pyflyt(
    action_6: np.ndarray | torch.Tensor,
    *,
    hover_thrust: float = DEFAULT_HOVER_THRUST,
    rate_scale: float = 2.0,
    xy_scale: float = 3.5,
    alt_scale: float = 0.6,
) -> np.ndarray:
    """Map one AeroJEPA action ``[dx, dy, d_alt, d_yaw, d_pitch, d_roll]`` to PyFlyt.

    PyFlyt QuadX expects ``(vp, vq, vr, T)`` - body angular-rate setpoints plus
    collective thrust in ``[0, 0.8]``.

    Empirically on ``QuadX-Hover-v4`` (euler): ``+vq`` moves ``+x``, ``-vp``
    moves ``+y``. Lateral AeroJEPA deltas are therefore mapped into lean rates;
    attitude deltas add a smaller residual.
    """
    a = np.asarray(action_6, dtype=np.float32).reshape(-1)
    if a.shape[0] != 6:
        raise ValueError(f"expected 6-DoF action, got shape {a.shape}")
    dx, dy, d_alt, d_yaw, d_pitch, d_roll = (float(x) for x in a)
    pi = float(np.pi)
    vp = float(np.clip((-dy * xy_scale) + (d_roll * rate_scale), -pi, pi))
    vq = float(np.clip((dx * xy_scale) + (d_pitch * rate_scale), -pi, pi))
    vr = float(np.clip(d_yaw * rate_scale, -pi, pi))
    thrust = float(np.clip(hover_thrust + alt_scale * d_alt, 0.0, 0.8))
    return np.array([vp, vq, vr, thrust], dtype=np.float32)


def _obs_xyz_vxyz(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract linear position / velocity from PyFlyt euler observation."""
    o = np.asarray(obs, dtype=np.float32).reshape(-1)
    lin_vel = o[6:9].copy()
    lin_pos = o[9:12].copy()
    return lin_pos, lin_vel


def _render_frame(env: Any, img_size: int) -> np.ndarray:
    """PyFlyt RGBA render → ``(3, H, W)`` float32 in ``[0, 1]``."""
    frame = env.render()
    if frame is None:
        raise RuntimeError("env.render() returned None; create the env with render_mode='rgb_array'")
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    if frame.shape[0] != img_size or frame.shape[1] != img_size:
        h, w = frame.shape[:2]
        y0 = max(0, (h - img_size) // 2)
        x0 = max(0, (w - img_size) // 2)
        frame = frame[y0 : y0 + img_size, x0 : x0 + img_size]
        if frame.shape[0] != img_size or frame.shape[1] != img_size:
            from PIL import Image

            frame = np.asarray(
                Image.fromarray(frame).resize((img_size, img_size), Image.BILINEAR)
            )
    out = np.ascontiguousarray(frame).astype(np.float32) / 255.0
    return np.transpose(out, (2, 0, 1))


def _assist_thrust(
    base_action: np.ndarray,
    obs: np.ndarray,
    z_des: float,
    *,
    kp: float = 0.35,
    kd: float = 0.15,
) -> np.ndarray:
    """Optional altitude PD on top of the mapped thrust (keeps demos aloft)."""
    pos, vel = _obs_xyz_vxyz(obs)
    action = base_action.copy()
    action[3] = float(
        np.clip(action[3] + kp * (z_des - float(pos[2])) - kd * float(vel[2]), 0.0, 0.8)
    )
    return action


def _brake_action(
    obs: np.ndarray,
    *,
    hover_thrust: float = DEFAULT_HOVER_THRUST,
    kd_xy: float = 0.9,
    kd_z: float = 0.15,
) -> np.ndarray:
    """Velocity-kill PD: lean against current velocity (no position term)."""
    pos, vel = _obs_xyz_vxyz(obs)
    pi = float(np.pi)
    # Same sign convention as _seek_action: +vq→+x, -vp→+y.
    vp = float(np.clip(kd_xy * vel[1], -pi, pi))
    vq = float(np.clip(-kd_xy * vel[0], -pi, pi))
    thrust = float(np.clip(hover_thrust - kd_z * vel[2], 0.0, 0.8))
    return np.array([vp, vq, 0.0, thrust], dtype=np.float32)


def _seek_action(
    obs: np.ndarray,
    goal_world: np.ndarray,
    *,
    hover_thrust: float = DEFAULT_HOVER_THRUST,
    kp_xy: float = 1.0,
    kd_xy: float = 0.55,
    kp_z: float = 0.35,
    kd_z: float = 0.15,
) -> np.ndarray:
    """Reactive PD that leans toward ``goal_world`` (no world model)."""
    pos, vel = _obs_xyz_vxyz(obs)
    err = np.asarray(goal_world, dtype=np.float32) - pos
    # Soften gains near the target to avoid overshoot / flip at the end.
    dist_xy = float(np.linalg.norm(err[:2]))
    gain = float(np.clip(dist_xy / 0.6, 0.25, 1.0))
    pi = float(np.pi)
    # Same sign convention as aerojepa_to_pyflyt: +vq→+x, -vp→+y.
    vp = float(np.clip(-(gain * kp_xy * err[1] - kd_xy * vel[1]), -pi, pi))
    vq = float(np.clip(gain * kp_xy * err[0] - kd_xy * vel[0], -pi, pi))
    thrust = float(np.clip(hover_thrust + kp_z * err[2] - kd_z * vel[2], 0.0, 0.8))
    return np.array([vp, vq, 0.0, thrust], dtype=np.float32)


def _make_constant_wind_fn(
    wind_vec: tuple[float, float, float] | np.ndarray,
    *,
    onset_seconds: float = 0.0,
) -> Any:
    """Build a PyFlyt wind-field callable: constant velocity after ``onset_seconds``."""
    vec = np.asarray(wind_vec, dtype=np.float64).reshape(3)

    def wind_fn(time: float, position: np.ndarray) -> np.ndarray:
        if time < onset_seconds:
            return np.zeros_like(position, dtype=np.float64)
        return np.broadcast_to(vec, position.shape).copy()

    return wind_fn


def _register_wind(env: Any, wind_fn: Any) -> None:
    """Attach a wind field to the live PyFlyt aviary (after ``reset``)."""
    aviary = getattr(getattr(env, "unwrapped", env), "env", None)
    if aviary is None or not hasattr(aviary, "register_wind_field_function"):
        raise RuntimeError(
            "PyFlyt env has no register_wind_field_function; cannot run wind_gust."
        )
    aviary.register_wind_field_function(wind_fn)


def classify_failure_mode(
    *,
    task: str,
    survived: bool,
    terminated: bool,
    truncated: bool,
    hit_max_steps: bool,
    final_altitude: float,
    max_xy_drift: float,
    reached: bool | None,
    recovered: bool | None,
    waypoints_reached: int = 0,
    waypoints_total: int = 0,
    wind_mps: float | None = None,
    wind_drift_fail: float = DEFAULT_WIND_DRIFT_FAIL,
    altitude_collapse: float = DEFAULT_ALTITUDE_COLLAPSE,
    flight_dome_size: float = 8.0,
) -> tuple[str, str]:
    """Return ``(failure_mode, detail)`` - ``ok`` when the episode succeeds."""
    # Ground contact vs dome/flip: both set terminated, but z tells them apart.
    if terminated and final_altitude <= altitude_collapse:
        return "crash", f"ground contact (z={final_altitude:.2f})"
    if terminated and max_xy_drift >= 0.85 * flight_dome_size:
        return (
            "out_of_bounds",
            f"hit flight dome (max_xy={max_xy_drift:.2f} m, dome≈{flight_dome_size} m)",
        )
    if terminated:
        return (
            "flip_or_unstable",
            f"terminated while aloft (z={final_altitude:.2f}, max_xy={max_xy_drift:.2f})",
        )
    if final_altitude < altitude_collapse:
        return "altitude_collapse", f"z={final_altitude:.2f} < {altitude_collapse}"
    if truncated and not hit_max_steps:
        return "out_of_bounds", f"truncated (max_xy={max_xy_drift:.2f} m)"

    if task == "wind_gust":
        if max_xy_drift > wind_drift_fail:
            return (
                "excessive_drift",
                f"max_xy={max_xy_drift:.2f} m under {wind_mps} m/s wind "
                f"(limit {wind_drift_fail} m)",
            )
        if hit_max_steps and survived:
            return "ok", f"held station under {wind_mps} m/s wind (max_xy={max_xy_drift:.2f} m)"
        return "ok", "survived wind gust"

    if task == "aggressive_turn":
        if waypoints_reached >= waypoints_total and waypoints_total > 0:
            return "ok", f"cleared {waypoints_reached}/{waypoints_total} legs of the L-turn"
        if waypoints_reached == 0:
            return "missed_turn", "never reached the first leg (failed to commit)"
        return (
            "missed_turn",
            f"cleared {waypoints_reached}/{waypoints_total} legs - broke at the corner",
        )

    if task == "waypoint":
        if reached:
            return "ok", "reached waypoint"
        return "missed_waypoint", f"final distance still large (max_xy={max_xy_drift:.2f})"

    if task == "recover":
        if recovered:
            return "ok", "recovered after kick"
        return "failed_recovery", f"final_xy={max_xy_drift:.2f} m (never re-acquired home)"

    if task == "hover":
        if survived and max_xy_drift < wind_drift_fail:
            return "ok", "hover hold"
        if not survived:
            return "crash", "hover episode ended early"
        return "excessive_drift", f"hover drifted max_xy={max_xy_drift:.2f} m"

    if survived:
        return "ok", "episode completed"
    return "unknown", "episode ended without survival"


@dataclass
class EpisodeResult:
    """One closed-loop episode with comparable metrics."""

    policy: str
    seed: int
    task: str
    steps: int
    total_reward: float
    mean_reward: float
    survived: bool
    final_altitude: float
    altitude_mae: float
    xy_drift: float
    max_xy_drift: float
    goal: tuple[float, float, float] | None = None
    final_goal_distance: float | None = None
    min_goal_distance: float | None = None
    reached: bool | None = None
    recovered: bool | None = None
    recovery_steps: int | None = None
    post_disturb_max_xy: float | None = None
    disturb_at: int | None = None
    disturb_steps: int | None = None
    mean_plan_cost: float | None = None
    planner_mode: str | None = None
    failure_mode: str | None = None
    failure_detail: str | None = None
    wind_mps: float | None = None
    wind_onset: int | None = None
    waypoints_reached: int | None = None
    waypoints_total: int | None = None
    frames_chw: list[np.ndarray] = field(default_factory=list, repr=False)
    altitudes: list[float] = field(default_factory=list)
    xy_drifts: list[float] = field(default_factory=list)
    goal_distances: list[float] = field(default_factory=list)
    positions_xyz: list[list[float]] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    phase_labels: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        d = asdict(self)
        for key in (
            "frames_chw",
            "altitudes",
            "xy_drifts",
            "goal_distances",
            "positions_xyz",
            "rewards",
            "phase_labels",
        ):
            d.pop(key, None)
        return d


@dataclass
class ClosedLoopDemoOutput:
    results: dict[str, EpisodeResult]
    metrics_path: Path | None
    plot_path: Path | None
    gif_paths: dict[str, Path]


def run_closed_loop_episode(
    model: AeroJEPA | None,
    device: torch.device,
    *,
    policy: PolicyName = "planner",
    task: str = "hover",
    img_size: int = 64,
    max_steps: int = 200,
    seed: int = 0,
    context_frames: int | None = None,
    horizon: int | None = None,
    num_candidates: int = 48,
    action_scale: float = 0.04,
    replan_every: int = 4,
    goal: tuple[float, float, float] | None = None,
    reach_threshold: float = DEFAULT_REACH_THRESHOLD,
    stop_on_reach: bool = True,
    hover_thrust: float = DEFAULT_HOVER_THRUST,
    assist_altitude: bool = True,
    record_frames: bool = True,
    env_id: str = DEFAULT_ENV_ID,
    agent_hz: int = 40,
    flight_dome_size: float = 5.0,
    max_duration_seconds: float = 12.0,
    disturb_at: int = DEFAULT_DISTURB_AT,
    disturb_steps: int = DEFAULT_DISTURB_STEPS,
    disturb_action: tuple[float, float, float, float] = DEFAULT_DISTURB_ACTION,
    recover_xy_threshold: float = DEFAULT_RECOVER_XY_THRESHOLD,
    min_kick_xy: float = DEFAULT_MIN_KICK_XY,
    damp_steps: int = DEFAULT_DAMP_STEPS,
    residual_head: ActionResidualHead | None = None,
    planner_mode: str = "shooting",
    grad_steps: int = 20,
    grad_lr: float = 0.06,
    grad_candidates: int = 12,
    grad_action_limit: float = 0.2,
    grad_vel_gain: float = 1.0,
    latent_smooth: float = 0.0,
    latent_refine_steps: int = 8,
    recover_seek_blend: float = 0.70,
    wind_mps: float = DEFAULT_WIND_MPS,
    wind_onset: int = DEFAULT_WIND_ONSET,
    wind_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    wind_drift_fail: float = DEFAULT_WIND_DRIFT_FAIL,
    aggressive_leg1: tuple[float, float, float] = DEFAULT_AGGRESSIVE_LEG1,
    aggressive_leg2: tuple[float, float, float] = DEFAULT_AGGRESSIVE_LEG2,
    aggressive_seek_blend_leg1: float = DEFAULT_AGGRESSIVE_SEEK_BLEND_LEG1,
    aggressive_seek_blend_leg2: float = DEFAULT_AGGRESSIVE_SEEK_BLEND_LEG2,
    corner_lookahead: float = DEFAULT_CORNER_LOOKAHEAD,
) -> EpisodeResult:
    """Run one PyFlyt episode under ``policy`` and return metrics (+ optional RGB).

    ``task``:
      * ``hover`` / ``waypoint`` / ``smoothness`` - LatentPlanner cost names
      * ``recover`` - settle, apply a lateral kick, then return toward start
        (planner uses waypoint cost on remaining displacement home)
      * ``wind_gust`` - hover hold under a moderate constant wind field
        (same heuristic action map; stress-tests station-keeping)
      * ``aggressive_turn`` - L-shaped two-leg course that forces a sharp 90° turn
    """
    known = set(COST_FUNCTIONS) | {"recover"} | STRESS_TASKS
    if task not in known:
        raise ValueError(f"task must be one of {sorted(known)}; got {task!r}")
    if policy == "planner" and model is None:
        raise ValueError("policy='planner' requires a model")

    if task == "waypoint" and goal is None:
        goal = DEFAULT_WAYPOINT_GOAL
    if task == "aggressive_turn" and goal is None:
        # Final corner of the L; legs are absolute displacements from start.
        goal = aggressive_leg2
    if task in ("waypoint", "recover", "aggressive_turn") and action_scale <= 0.05:
        action_scale = 0.10
    if task == "aggressive_turn":
        if action_scale < 0.10:
            action_scale = 0.10
        # Slightly longer horizon helps the corner; must fit in model T
        # (default T=8 → context=4 + horizon=4).
        if horizon is None:
            horizon = 4
        # Tip-in at the corner rarely tags the tight hover threshold.
        if reach_threshold <= DEFAULT_REACH_THRESHOLD + 1e-9:
            reach_threshold = DEFAULT_AGGRESSIVE_REACH

    # Planner cost: recover / aggressive_turn home via waypoint scoring; wind uses hover.
    if task in ("recover", "aggressive_turn"):
        planner_cost = "waypoint"
    elif task == "wind_gust":
        planner_cost = "hover"
    else:
        planner_cost = task

    # Corner / disturbance-aware cost weights for the gradient planner.
    if task == "aggressive_turn":
        cost_weights = MultiStepCostWeights(
            pos_terminal=1.2,
            pos_running=0.15,
            vel_terminal=1.2,  # arrive slow - kills corner overshoot
            vel_running=0.15,
            attitude=0.5,
            attitude_rate=0.25,
            effort=0.05,
            latent_smooth=latent_smooth,
        )
    elif task == "wind_gust":
        cost_weights = MultiStepCostWeights(
            pos_terminal=1.0,
            pos_running=0.4,
            vel_terminal=1.0,
            vel_running=0.2,
            attitude=0.4,
            attitude_rate=0.15,
            effort=0.02,
            latent_smooth=latent_smooth,
        )
    else:
        cost_weights = MultiStepCostWeights(latent_smooth=latent_smooth)

    if task == "recover":
        # Give the knock room to displace without immediate out-of-dome failure.
        flight_dome_size = max(flight_dome_size, 8.0)
        max_duration_seconds = max(max_duration_seconds, 20.0)
    if task in STRESS_TASKS:
        flight_dome_size = max(flight_dome_size, 8.0)
        max_duration_seconds = max(max_duration_seconds, 20.0)

    env = make_pyflyt_env(
        env_id,
        render_mode="rgb_array",
        angle_representation="euler",
        render_resolution=(img_size, img_size),
        flight_dome_size=flight_dome_size,
        max_duration_seconds=max_duration_seconds,
        agent_hz=agent_hz,
    )

    planner: LatentPlanner | None = None
    if policy == "planner":
        assert model is not None
        num_temporal = model.encoder.num_temporal
        if context_frames is None:
            context_frames = max(1, num_temporal // 2)
        context_frames = max(1, min(int(context_frames), num_temporal - 1))
        if horizon is not None and context_frames + int(horizon) > num_temporal:
            # Prefer the requested prediction horizon; shrink context to fit T.
            horizon = min(int(horizon), num_temporal - 1)
            context_frames = max(1, num_temporal - int(horizon))
        planner = LatentPlanner(
            model,
            device,
            cost_fn=planner_cost,
            residual_head=residual_head,
            planning=planner_mode,
            grad_steps=grad_steps,
            grad_lr=grad_lr,
            grad_action_limit=grad_action_limit,
            latent_refine_steps=latent_refine_steps,
            cost_weights=cost_weights,
        )

    rng = np.random.default_rng(seed)
    obs, _info = env.reset(seed=seed)
    start_pos, _ = _obs_xyz_vxyz(obs)
    z0 = float(start_pos[2])
    goal_disp = np.zeros(3, dtype=np.float32) if goal is None else np.asarray(goal, dtype=np.float32)
    # Waypoint / aggressive final goal = start+goal; recover always homes to start.
    goal_world = start_pos.copy() if task == "recover" else (start_pos + goal_disp)
    z_des = float(goal_world[2]) if task in ("waypoint", "recover", "aggressive_turn") else z0
    kick = np.asarray(disturb_action, dtype=np.float32)

    # L-turn waypoints in world frame (leg1 then leg2 / corner).
    turn_waypoints: list[np.ndarray] = []
    if task == "aggressive_turn":
        turn_waypoints = [
            start_pos + np.asarray(aggressive_leg1, dtype=np.float32),
            start_pos + np.asarray(aggressive_leg2, dtype=np.float32),
        ]
        goal_world = turn_waypoints[0].copy()
        z_des = float(goal_world[2])

    # Moderate wind: register after reset so the aviary is live. Same heuristic
    # action map as every other task - wind is the only new stressor.
    wind_vec = None
    if task == "wind_gust":
        direction = np.asarray(wind_direction, dtype=np.float64).reshape(3)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-8:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            norm = 1.0
        wind_vec = (direction / norm) * float(wind_mps)
        onset_s = float(wind_onset) / float(agent_hz)
        _register_wind(env, _make_constant_wind_fn(wind_vec, onset_seconds=onset_s))

    buffer: deque[np.ndarray] = deque(maxlen=context_frames or 1)
    first_frame = _render_frame(env, img_size)
    buffer.append(first_frame)

    pending: list[np.ndarray] = []
    plan_costs: list[float] = []
    frames_out: list[np.ndarray] = [first_frame] if record_frames else []
    altitudes: list[float] = [z0]
    xy_drifts: list[float] = [0.0]
    goal_distances: list[float] = [float(np.linalg.norm(start_pos - goal_world))]
    positions_xyz: list[list[float]] = [start_pos.tolist()]
    rewards: list[float] = []
    phase_labels: list[str] = ["settle"]
    total = 0.0
    survived = True
    reached = False
    recovered = False
    recovery_steps: int | None = None
    post_disturb_max_xy = 0.0
    disturb_end = disturb_at + disturb_steps
    damp_end = disturb_end + damp_steps
    waypoints_reached = 0
    terminated = False
    truncated = False
    hit_max_steps = False

    damp_max_end = disturb_end + max(damp_steps, DEFAULT_DAMP_MAX_STEPS)

    try:
        for step in range(max_steps):
            if task == "recover" and disturb_at <= step < disturb_end:
                phase = "kick"
            elif task == "recover" and disturb_end <= step < damp_end:
                phase = "damp"
            elif task == "recover" and step >= damp_end:
                # Adaptive braking: stay in damp while the craft still carries
                # kick momentum (coming home hot is what tips it over).
                _, vel_now = _obs_xyz_vxyz(obs)
                speed_xy = float(np.linalg.norm(vel_now[:2]))
                if step < damp_max_end and speed_xy > DEFAULT_BRAKE_VEL:
                    phase = "damp"
                else:
                    phase = "recover"
            elif task == "wind_gust" and step >= wind_onset:
                phase = "wind"
            elif task == "aggressive_turn":
                phase = "leg1" if waypoints_reached == 0 else "leg2"
            else:
                phase = "settle"

            if phase == "kick":
                action = kick.copy()
                pending.clear()  # force replan after the knock
            elif phase == "damp":
                # Actively kill lateral momentum (lean against velocity) rather
                # than passively hovering - passive coast leaves the craft too
                # hot for any recovery policy to bring home upright.
                action = _brake_action(obs, hover_thrust=hover_thrust)
                pending.clear()
            elif policy == "planner":
                assert planner is not None and context_frames is not None
                if not pending:
                    while len(buffer) < context_frames:
                        buffer.appendleft(buffer[0].copy())
                    ctx = torch.from_numpy(np.stack(list(buffer)[-context_frames:]))
                    pos_now, vel_now = _obs_xyz_vxyz(obs)
                    remaining = None
                    if task == "aggressive_turn" and turn_waypoints:
                        # Corner look-ahead: before tagging leg1, start aiming at
                        # leg2 so the plan commits to the 90° turn early.
                        target = goal_world
                        if waypoints_reached == 0 and len(turn_waypoints) >= 2:
                            d_leg1 = float(np.linalg.norm(pos_now - turn_waypoints[0]))
                            if d_leg1 <= float(corner_lookahead):
                                target = turn_waypoints[1]
                        remaining = tuple(float(x) for x in (target - pos_now))
                    elif task in ("waypoint", "recover"):
                        remaining = tuple(float(x) for x in (goal_world - pos_now))
                    elif goal is not None:
                        remaining = goal
                    n_cand = (
                        grad_candidates if planner_mode == "gradient" else num_candidates
                    )
                    # Momentum-aware braking: hand the gradient planner the current
                    # velocity so it can plan actions that arrest drift, not just
                    # chase the position goal (crucial after a disturbance kick).
                    init_vel = None
                    if planner_mode == "gradient" and grad_vel_gain > 0:
                        init_vel = tuple(float(x) for x in (vel_now * grad_vel_gain))
                    result = planner.plan(
                        ctx,
                        num_candidates=n_cand,
                        horizon=horizon,
                        action_scale=action_scale,
                        goal=remaining,
                        seed=seed + step,
                        init_velocity=init_vel,
                    )
                    plan_costs.append(float(result.costs[result.best_index]))
                    n_exec = max(1, min(replan_every, int(result.horizon)))
                    best_lat = result.best_latents  # (H, S, D)
                    for i in range(n_exec):
                        mapped = map_aero_with_optional_residual(
                            result.best_actions[i].numpy(),
                            residual_head=residual_head,
                            latent=best_lat[i],
                            hover_thrust=hover_thrust,
                        )
                        pending.append(mapped)
                action = pending.pop(0)
                # Blend a light seek PD during recover so the heuristic action
                # map cannot fling the drone after the kick (research demo
                # stabilizer - not a claim that planning alone is sufficient).
                # Lower ``recover_seek_blend`` to attribute more of the recovery
                # to the planner itself (used to compare planner modes).
                if task == "recover" and phase == "recover":
                    seek = _seek_action(obs, goal_world, hover_thrust=hover_thrust)
                    blend = float(np.clip(recover_seek_blend, 0.0, 1.0))
                    action = ((1.0 - blend) * action + blend * seek).astype(np.float32)
                elif task == "aggressive_turn":
                    # Light seek PD (stronger on leg2) keeps tip-over in check at
                    # the corner while the planner still owns most of the command.
                    seek_goal = goal_world
                    if waypoints_reached == 0 and len(turn_waypoints) >= 2:
                        pos_now, _ = _obs_xyz_vxyz(obs)
                        if float(np.linalg.norm(pos_now - turn_waypoints[0])) <= float(
                            corner_lookahead
                        ):
                            seek_goal = turn_waypoints[1]
                    seek = _seek_action(obs, seek_goal, hover_thrust=hover_thrust)
                    blend = (
                        aggressive_seek_blend_leg2
                        if waypoints_reached >= 1
                        else aggressive_seek_blend_leg1
                    )
                    blend = float(np.clip(blend, 0.0, 1.0))
                    if blend > 0:
                        action = ((1.0 - blend) * action + blend * seek).astype(
                            np.float32
                        )
            elif policy == "hover":
                action = np.array([0.0, 0.0, 0.0, hover_thrust], dtype=np.float32)
            elif policy == "seek":
                action = _seek_action(obs, goal_world, hover_thrust=hover_thrust)
            elif policy == "random":
                low = env.action_space.low.astype(np.float32)
                high = env.action_space.high.astype(np.float32)
                action = (rng.uniform(low, high) * 0.35).astype(np.float32)
            elif policy == "inert":
                action = np.zeros(4, dtype=np.float32)
            else:  # pragma: no cover
                raise ValueError(f"unknown policy {policy!r}")

            if assist_altitude and policy in ("planner", "hover", "random", "seek"):
                action = _assist_thrust(action, obs, z_des)

            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            rewards.append(float(reward))
            phase_labels.append(phase)

            pos, _vel = _obs_xyz_vxyz(obs)
            xy = float(np.linalg.norm(pos[:2] - start_pos[:2]))
            altitudes.append(float(pos[2]))
            xy_drifts.append(xy)
            dist = float(np.linalg.norm(pos - goal_world))
            goal_distances.append(dist)
            positions_xyz.append(pos.tolist())
            if dist <= reach_threshold:
                reached = True

            # Advance L-turn waypoint when the current leg is reached.
            if task == "aggressive_turn" and waypoints_reached < len(turn_waypoints):
                if dist <= reach_threshold:
                    waypoints_reached += 1
                    pending.clear()
                    if waypoints_reached < len(turn_waypoints):
                        goal_world = turn_waypoints[waypoints_reached].copy()
                        z_des = float(goal_world[2])
                        reached = False
                    else:
                        reached = True

            if task == "recover" and step >= disturb_at:
                post_disturb_max_xy = max(post_disturb_max_xy, xy)
            if task == "recover" and step >= damp_end:
                # Only count recovery after a real knock (peak XY above min_kick_xy).
                # Require still aloft (z>0.25); do not demand exact return to z0 -
                # altitude PD is approximate and recovery is primarily an XY task.
                if (
                    not recovered
                    and post_disturb_max_xy >= min_kick_xy
                    and xy <= recover_xy_threshold
                    and float(pos[2]) > 0.25
                ):
                    recovered = True
                    recovery_steps = step - damp_end + 1

            frame = _render_frame(env, img_size)
            buffer.append(frame)
            if record_frames:
                frames_out.append(frame)

            if reached and stop_on_reach and task in ("waypoint", "aggressive_turn"):
                survived = True
                break

            if recovered and task == "recover" and stop_on_reach:
                survived = True
                break

            if terminated or truncated:
                survived = not terminated
                break
        else:
            hit_max_steps = True
    finally:
        env.close()

    alt_target = np.full(len(altitudes), z_des, dtype=np.float32)
    altitude_mae = float(np.mean(np.abs(np.asarray(altitudes) - alt_target)))
    failure_mode, failure_detail = classify_failure_mode(
        task=task,
        survived=survived,
        terminated=bool(terminated),
        truncated=bool(truncated),
        hit_max_steps=hit_max_steps,
        final_altitude=float(altitudes[-1]),
        max_xy_drift=float(max(xy_drifts)),
        reached=reached if task in ("waypoint", "aggressive_turn") else None,
        recovered=recovered if task == "recover" else None,
        waypoints_reached=waypoints_reached if task == "aggressive_turn" else 0,
        waypoints_total=len(turn_waypoints) if task == "aggressive_turn" else 0,
        wind_mps=float(wind_mps) if task == "wind_gust" else None,
        wind_drift_fail=wind_drift_fail,
        flight_dome_size=float(flight_dome_size),
    )
    return EpisodeResult(
        policy=policy,
        seed=seed,
        task=task,
        steps=len(rewards),
        total_reward=total,
        mean_reward=float(total / max(len(rewards), 1)),
        survived=survived,
        final_altitude=float(altitudes[-1]),
        altitude_mae=altitude_mae,
        xy_drift=float(xy_drifts[-1]),
        max_xy_drift=float(max(xy_drifts)),
        goal=None if goal is None else tuple(float(x) for x in goal),
        final_goal_distance=float(goal_distances[-1]),
        min_goal_distance=float(min(goal_distances)),
        reached=reached if task in ("waypoint", "aggressive_turn") else None,
        recovered=recovered if task == "recover" else None,
        recovery_steps=recovery_steps if task == "recover" else None,
        post_disturb_max_xy=float(post_disturb_max_xy) if task == "recover" else None,
        disturb_at=disturb_at if task == "recover" else None,
        disturb_steps=disturb_steps if task == "recover" else None,
        mean_plan_cost=float(np.mean(plan_costs)) if plan_costs else None,
        planner_mode=planner_mode if policy == "planner" else None,
        failure_mode=failure_mode,
        failure_detail=failure_detail,
        wind_mps=float(wind_mps) if task == "wind_gust" else None,
        wind_onset=int(wind_onset) if task == "wind_gust" else None,
        waypoints_reached=waypoints_reached if task == "aggressive_turn" else None,
        waypoints_total=len(turn_waypoints) if task == "aggressive_turn" else None,
        frames_chw=frames_out,
        altitudes=altitudes,
        xy_drifts=xy_drifts,
        goal_distances=goal_distances,
        positions_xyz=positions_xyz,
        rewards=rewards,
        phase_labels=phase_labels,
    )


def _save_policy_gif(
    frames_chw: list[np.ndarray],
    out_path: Path,
    *,
    label: str,
    phase_labels: list[str] | None = None,
    upscale: int = 4,
    duration_ms: int = 50,
    stride: int = 2,
) -> Path:
    from PIL import Image, ImageDraw

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    phase_color = {
        "settle": (54, 100, 227),
        "kick": (200, 50, 50),
        "damp": (120, 80, 160),
        "recover": (230, 126, 34),
        "wind": (40, 140, 180),
        "leg1": (54, 100, 227),
        "leg2": (180, 80, 40),
    }
    images = []
    phases = phase_labels or ["settle"] * len(frames_chw)
    for i, chw in enumerate(frames_chw[::stride]):
        idx = min(i * stride, len(phases) - 1)
        phase = phases[idx]
        arr = (np.clip(chw.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(arr).resize(
            (arr.shape[1] * upscale, arr.shape[0] * upscale), Image.NEAREST
        )
        draw = ImageDraw.Draw(img)
        bar_h = max(14, img.height // 10)
        draw.rectangle([0, 0, img.width, bar_h], fill=phase_color.get(phase, (40, 40, 40)))
        draw.text((6, 2), f"{label}  {phase}  t={i * stride}", fill=(240, 240, 240))
        images.append(img)
    if not images:
        raise ValueError("no frames to save")
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )
    return out_path


def _save_comparison_plot(
    results: dict[str, EpisodeResult],
    out_path: Path,
    *,
    task: str,
    goal: tuple[float, float, float] | None,
    disturb_at: int | None = None,
    disturb_steps: int | None = None,
    wind_onset: int | None = None,
    aggressive_legs: tuple[tuple[float, float, float], tuple[float, float, float]] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if task in ("waypoint", "aggressive_turn"):
        fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
        for name, ep in results.items():
            xyz = np.asarray(ep.positions_xyz, dtype=np.float32)
            t = np.arange(len(ep.goal_distances))
            axes[0].plot(xyz[:, 0], xyz[:, 1], label=name, linewidth=1.8)
            axes[1].plot(t, ep.goal_distances, label=name, linewidth=1.8)
            axes[2].plot(t, ep.altitudes, label=name, linewidth=1.8)
        if results:
            start = np.asarray(next(iter(results.values())).positions_xyz[0], dtype=np.float32)
            axes[0].scatter([start[0]], [start[1]], c="k", s=40, zorder=5, label="start")
            if task == "aggressive_turn" and aggressive_legs is not None:
                g1 = start[:2] + np.asarray(aggressive_legs[0][:2], dtype=np.float32)
                g2 = start[:2] + np.asarray(aggressive_legs[1][:2], dtype=np.float32)
                axes[0].scatter([g1[0]], [g1[1]], c="orange", marker="o", s=70, zorder=5, label="leg1")
                axes[0].scatter([g2[0]], [g2[1]], c="red", marker="*", s=90, zorder=5, label="leg2")
                axes[0].plot([start[0], g1[0], g2[0]], [start[1], g1[1], g2[1]], "k--", linewidth=1, alpha=0.5)
            elif goal is not None:
                g = start[:2] + np.asarray(goal[:2], dtype=np.float32)
                axes[0].scatter([g[0]], [g[1]], c="red", marker="*", s=90, zorder=5, label="goal")
            axes[1].axhline(DEFAULT_REACH_THRESHOLD, color="gray", linestyle="--", linewidth=1)
        title = "XY path (L-turn)" if task == "aggressive_turn" else "XY path"
        axes[0].set_title(title)
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        axes[0].axis("equal")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)
        axes[1].set_title("Distance to active goal")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("meters")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)
        axes[2].set_title("Altitude")
        axes[2].set_xlabel("step")
        axes[2].set_ylabel("z")
        axes[2].legend(fontsize=7)
        axes[2].grid(True, alpha=0.3)
    elif task == "recover":
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
        kick_labeled = False
        for name, ep in results.items():
            t = np.arange(len(ep.altitudes))
            axes[0].plot(t, ep.altitudes, label=name, linewidth=1.8)
            axes[1].plot(t, ep.xy_drifts, label=name, linewidth=1.8)
        if disturb_at is not None and disturb_steps is not None:
            for ax in axes:
                ax.axvspan(
                    disturb_at,
                    disturb_at + disturb_steps,
                    color="red",
                    alpha=0.15,
                    label="kick" if not kick_labeled else None,
                )
                kick_labeled = True
            axes[1].axhline(DEFAULT_RECOVER_XY_THRESHOLD, color="gray", linestyle="--", linewidth=1)
        axes[0].set_title("Altitude (recover)")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("z")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)
        axes[1].set_title("XY drift from start")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("meters")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)
    elif task == "wind_gust":
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
        wind_labeled = False
        for name, ep in results.items():
            t = np.arange(len(ep.altitudes))
            axes[0].plot(t, ep.altitudes, label=name, linewidth=1.8)
            axes[1].plot(t, ep.xy_drifts, label=name, linewidth=1.8)
        if wind_onset is not None:
            for ax in axes:
                ax.axvline(
                    wind_onset,
                    color="teal",
                    linestyle="--",
                    linewidth=1.2,
                    label="wind on" if not wind_labeled else None,
                )
                wind_labeled = True
            axes[1].axhline(DEFAULT_WIND_DRIFT_FAIL, color="gray", linestyle="--", linewidth=1)
        axes[0].set_title("Altitude (wind gust)")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("z")
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)
        axes[1].set_title("XY drift under wind")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("meters")
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
        for name, ep in results.items():
            t = np.arange(len(ep.altitudes))
            axes[0].plot(t, ep.altitudes, label=name, linewidth=1.8)
            axes[1].plot(t, ep.xy_drifts, label=name, linewidth=1.8)
        axes[0].set_title("Altitude")
        axes[0].set_xlabel("step")
        axes[0].set_ylabel("z")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)
        axes[1].set_title("XY drift from start")
        axes[1].set_xlabel("step")
        axes[1].set_ylabel("meters")
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def run_closed_loop_demo(
    model: AeroJEPA,
    device: torch.device,
    *,
    policies: tuple[PolicyName, ...] | None = None,
    out_dir: str | Path = "visualizations/closed_loop",
    img_size: int = 64,
    max_steps: int = 160,
    seed: int = 0,
    task: str = "hover",
    goal: tuple[float, float, float] | None = None,
    assist_altitude: bool = True,
    reach_threshold: float = DEFAULT_REACH_THRESHOLD,
    stop_on_reach: bool = True,
    disturb_at: int = DEFAULT_DISTURB_AT,
    disturb_steps: int = DEFAULT_DISTURB_STEPS,
    recover_xy_threshold: float = DEFAULT_RECOVER_XY_THRESHOLD,
    residual_head: ActionResidualHead | None = None,
    planner_mode: str = "shooting",
    grad_steps: int = 20,
    grad_lr: float = 0.06,
    grad_candidates: int = 12,
    grad_action_limit: float = 0.2,
    grad_vel_gain: float = 1.0,
    latent_smooth: float = 0.0,
    latent_refine_steps: int = 8,
    recover_seek_blend: float = 0.70,
    wind_mps: float = DEFAULT_WIND_MPS,
    wind_onset: int = DEFAULT_WIND_ONSET,
    wind_drift_fail: float = DEFAULT_WIND_DRIFT_FAIL,
    aggressive_leg1: tuple[float, float, float] = DEFAULT_AGGRESSIVE_LEG1,
    aggressive_leg2: tuple[float, float, float] = DEFAULT_AGGRESSIVE_LEG2,
    aggressive_seek_blend_leg1: float = DEFAULT_AGGRESSIVE_SEEK_BLEND_LEG1,
    aggressive_seek_blend_leg2: float = DEFAULT_AGGRESSIVE_SEEK_BLEND_LEG2,
    corner_lookahead: float = DEFAULT_CORNER_LOOKAHEAD,
    **episode_kwargs: Any,
) -> ClosedLoopDemoOutput:
    """Run several policies, write metrics JSON, trajectory plot, and GIFs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if policies is None:
        if task in ("waypoint", "recover", "aggressive_turn"):
            policies = ("planner", "seek", "hover", "random")
        elif task == "wind_gust":
            policies = ("planner", "hover", "random")
        else:
            policies = ("planner", "hover", "random")
    if task == "waypoint" and goal is None:
        goal = DEFAULT_WAYPOINT_GOAL
    if task == "aggressive_turn" and goal is None:
        goal = aggressive_leg2

    results: dict[str, EpisodeResult] = {}
    gif_paths: dict[str, Path] = {}
    for policy in policies:
        ep = run_closed_loop_episode(
            model if policy == "planner" else None,
            device,
            policy=policy,
            task=task,
            img_size=img_size,
            max_steps=max_steps,
            seed=seed,
            goal=goal,
            assist_altitude=assist_altitude,
            reach_threshold=reach_threshold,
            stop_on_reach=stop_on_reach,
            disturb_at=disturb_at,
            disturb_steps=disturb_steps,
            recover_xy_threshold=recover_xy_threshold,
            residual_head=residual_head,
            planner_mode=planner_mode,
            grad_steps=grad_steps,
            grad_lr=grad_lr,
            grad_candidates=grad_candidates,
            grad_action_limit=grad_action_limit,
            grad_vel_gain=grad_vel_gain,
            latent_smooth=latent_smooth,
            latent_refine_steps=latent_refine_steps,
            recover_seek_blend=recover_seek_blend,
            wind_mps=wind_mps,
            wind_onset=wind_onset,
            wind_drift_fail=wind_drift_fail,
            aggressive_leg1=aggressive_leg1,
            aggressive_leg2=aggressive_leg2,
            aggressive_seek_blend_leg1=aggressive_seek_blend_leg1,
            aggressive_seek_blend_leg2=aggressive_seek_blend_leg2,
            corner_lookahead=corner_lookahead,
            record_frames=True,
            **episode_kwargs,
        )
        results[policy] = ep
        gif_paths[policy] = _save_policy_gif(
            ep.frames_chw,
            out_dir / f"closed_loop_{task}_{policy}.gif",
            label=policy,
            phase_labels=ep.phase_labels,
        )

    metrics = {name: ep.summary() for name, ep in results.items()}
    metrics_path = out_dir / f"closed_loop_{task}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    plot_path = _save_comparison_plot(
        results,
        out_dir / f"closed_loop_{task}_trajectories.png",
        task=task,
        goal=goal,
        disturb_at=disturb_at if task == "recover" else None,
        disturb_steps=disturb_steps if task == "recover" else None,
        wind_onset=wind_onset if task == "wind_gust" else None,
        aggressive_legs=(aggressive_leg1, aggressive_leg2)
        if task == "aggressive_turn"
        else None,
    )

    return ClosedLoopDemoOutput(
        results=results,
        metrics_path=metrics_path,
        plot_path=plot_path,
        gif_paths=gif_paths,
    )


def stitch_demo_reel(
    gif_specs: list[tuple[str, Path]],
    out_path: str | Path,
    *,
    panel_size: tuple[int, int] = (256, 256),
    hold_title_frames: int = 8,
    duration_ms: int = 60,
) -> Path:
    """Concatenate labeled closed-loop GIFs into one demo reel.

    ``gif_specs`` is a list of ``(section_title, gif_path)``. Each section gets a
    short title card, then the GIF frames (resized to ``panel_size``).
    """
    from PIL import Image, ImageDraw, ImageFont

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = panel_size
    frames: list[Image.Image] = []

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 22)
        font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_small = font

    for title, gif_path in gif_specs:
        gif_path = Path(gif_path)
        if not gif_path.exists():
            raise FileNotFoundError(gif_path)

        title_img = Image.new("RGB", (w, h), (18, 22, 30))
        draw = ImageDraw.Draw(title_img)
        draw.text((16, h // 2 - 20), title, fill=(240, 240, 240), font=font)
        draw.text((16, h // 2 + 12), "AeroJEPA closed-loop", fill=(160, 170, 190), font=font_small)
        for _ in range(hold_title_frames):
            frames.append(title_img.copy())

        src = Image.open(gif_path)
        try:
            while True:
                frame = src.convert("RGB").resize((w, h), Image.NEAREST)
                frames.append(frame)
                src.seek(src.tell() + 1)
        except EOFError:
            pass
        finally:
            src.close()

    if not frames:
        raise ValueError("no frames to stitch")
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return out_path
