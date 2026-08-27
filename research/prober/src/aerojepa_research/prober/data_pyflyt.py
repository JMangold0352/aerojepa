"""PyFlyt data generator for AeroProber.

Generates clips of ``(frames, actions, metric_state)`` where:
- ``frames``    (T, 3, H, W) float32 in [0, 1] -- rendered RGB from the drone cam.
- ``actions``   (T, 6) float32 -- AeroJEPA ACTION_COLUMNS convention
                 (dx, dy, d_altitude, d_yaw, d_pitch, d_roll), derived from
                 consecutive PyFlyt states so the frozen action-conditioned
                 predictor (if any) sees in-distribution inputs.
- ``metric_state`` (T, 12) float32 -- [pos_world(3), vel_world(3),
                    euler_att_deg(yaw,pitch,roll), ang_vel_body_pqr_deg(3)].
                    PyFlyt obs ``lin_vel``/``ang_vel`` are body-frame (rad/s for
                    rates); we convert to world velocity + deg/s body rates.

The clips are driven by random PyFlyt control actions (angular rates + thrust),
which produce rich, physically-plausible quadrotor motion. We record the
*consequence* of each control step as the metric state, and derive AeroJEPA-style
pose-delta actions from consecutive states so the frozen world model sees the
same kind of inputs it was trained on.

IMPORTANT: PyFlyt/PyBullet must run OUTSIDE the Cursor sandbox (it segfaults
inside). All scripts that import this module should be run with full permissions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# Suppress PyFlyt's verbose pybullet logging and the RGBA-render warning.
warnings.filterwarnings("ignore", message=".*RGB-array rendering.*")


# AeroJEPA action convention (see src/aerojepa/data/telemetry.py).
ACTION_DIM = 6
METRIC_STATE_DIM = 12  # pos(3) + vel(3) + att(3) + ang_vel(3)


def _wrap_degrees(delta: np.ndarray) -> np.ndarray:
    """Wrap angle differences to [-180, 180) -- matches telemetry.wrap_degrees."""
    return (delta + 180.0) % 360.0 - 180.0


def _euler_rad_to_deg_yaw_pitch_roll(ang_pos_rpy: np.ndarray) -> np.ndarray:
    """Convert PyBullet Euler (roll, pitch, yaw) radians to (yaw, pitch, roll) degrees.

    PyBullet returns attitude as (roll, pitch, yaw). AeroJEPA's telemetry uses
    (yaw, pitch, roll) in degrees. We reorder + convert to match the convention
    the frozen model was trained against (see telemetry.ACTION_COLUMNS order:
    d_yaw, d_pitch, d_roll).
    """
    roll, pitch, yaw = ang_pos_rpy  # radians
    return np.array([np.degrees(yaw), np.degrees(pitch), np.degrees(roll)], dtype=np.float32)


def _euler_ypr_deg_to_R(att_deg_ypr: np.ndarray) -> np.ndarray:
    """Body→world rotation from (yaw, pitch, roll) degrees. Returns (3, 3)."""
    y, p, r = np.deg2rad(att_deg_ypr.astype(np.float64))
    cy, sy = np.cos(y), np.sin(y)
    cp, sp = np.cos(p), np.sin(p)
    cr, sr = np.cos(r), np.sin(r)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float32,
    )


def states_to_actions(states: np.ndarray) -> np.ndarray:
    """Derive AeroJEPA 6-DoF pose-delta actions from consecutive metric states.

    Matches the convention in ``src/aerojepa/data/telemetry.py``:
    - Linear channels (dx, dy, d_altitude) = the per-frame **body velocity**
      (vgx/vgy/vgz), used directly -- NOT a velocity delta. The real AeroJEPA
      pipeline uses the reported body velocities as the linear action channels.
    - Angular channels (d_yaw, d_pitch, d_roll) = wrapped frame-to-frame
      attitude deltas (degrees).

    Parameters
    ----------
    states : (T, 12)  [pos, vel_world, euler_att_deg(yaw,pitch,roll), ang_vel_pqr_deg]

    Returns
    -------
    actions : (T, 6)  (dx, dy, d_altitude, d_yaw, d_pitch, d_roll)
              The first row's angular channels are zero (no previous frame);
              the linear channels carry body velocity (``R^T v_world``).
    """
    T = states.shape[0]
    actions = np.zeros((T, ACTION_DIM), dtype=np.float32)
    if T < 1:
        return actions
    vel_w = states[:, 3:6]
    att = states[:, 6:9]  # yaw, pitch, roll (deg)
    for t in range(T):
        R = _euler_ypr_deg_to_R(att[t])
        actions[t, 0:3] = R.T @ vel_w[t]  # body velocity for telemetry match
    if T >= 2:
        actions[1:, 3] = _wrap_degrees(np.diff(att[:, 0]))  # d_yaw
        actions[1:, 4] = np.diff(att[:, 1])  # d_pitch
        actions[1:, 5] = np.diff(att[:, 2])  # d_roll
    return actions


@dataclass
class PyFlytClip:
    """One generated clip.

    Attributes
    ----------
    frames : (T, 3, H, W) float32 in [0, 1]
    actions : (T, 6) float32
        AeroJEPA-convention pose-delta actions (state-derived). Kept for
        provenance and compatibility with the frozen world model, which was
        trained on this convention. NOT used as the prober's action input --
        see ``control_actions``.
    metric_state : (T, 12) float32
        Ground-truth [pos, vel, euler_att_deg, ang_vel].
    control_actions : (T, 4) float32
        Raw PyFlyt control commands (vp, vq, vr, T) -- angular-rate + thrust
        setpoints that drove the simulation. These are genuinely exogenous
        (sampled from a RNG), not derived from state, so they are the leak-free
        action input for the prober's integrator.
    """

    frames: torch.Tensor       # (T, 3, H, W) float32 in [0, 1]
    actions: torch.Tensor      # (T, 6) float32 -- AeroJEPA convention (state-derived)
    metric_state: torch.Tensor # (T, 12) float32
    control_actions: torch.Tensor  # (T, 4) float32 -- raw (vp, vq, vr, T)


def _normalize_actions(actions: np.ndarray, scale: float = 0.1) -> np.ndarray:
    """tanh-normalize actions to roughly unit range (matches telemetry helper)."""
    return np.tanh(actions / scale).astype(np.float32)


def generate_clip(
    seed: int,
    num_frames: int = 8,
    img_size: int = 64,
    flight_dome_size: float = 5.0,
    max_duration_seconds: float = 10.0,
    agent_hz: int = 40,
    wind_mps: float = 0.0,
    wind_direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
    control_mode: str = "random",
) -> PyFlytClip:
    """Render one deterministic PyFlyt clip with full metric state.

    The drone is driven by:

    * ``random`` — small random PyFlyt controls
    * ``hover`` — altitude PD + lean against XY velocity / known wind
    * ``kick`` — brief lateral disturb, then brake + home toward start
    * ``turn`` — seek along an L-path (leg1 then leg2) for corner supervision
    """
    import gymnasium
    import PyFlyt.gym_envs  # noqa: F401  -- registers envs

    if control_mode not in ("random", "hover", "kick", "turn"):
        raise ValueError(
            f"control_mode must be one of random/hover/kick/turn, got {control_mode!r}"
        )

    rng = np.random.default_rng(seed)
    env = gymnasium.make(
        "PyFlyt/QuadX-Hover-v4",
        render_mode="rgb_array",
        angle_representation="euler",
        render_resolution=(img_size, img_size),
        flight_dome_size=flight_dome_size,
        max_duration_seconds=max_duration_seconds,
        agent_hz=agent_hz,
    )
    try:
        obs, _info = env.reset(seed=seed)

        # Optional constant wind field (same API as closed_loop wind_gust).
        if wind_mps > 0:
            direction = np.asarray(wind_direction, dtype=np.float64).reshape(3)
            norm = float(np.linalg.norm(direction))
            if norm < 1e-8:
                direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                norm = 1.0
            wind_vec = (direction / norm) * float(wind_mps)

            def wind_fn(time: float, position: np.ndarray) -> np.ndarray:
                return np.broadcast_to(wind_vec, position.shape).copy()

            env.unwrapped.env.register_wind_field_function(wind_fn)
        else:
            wind_vec = np.zeros(3, dtype=np.float64)

        # Pre-sample random controls when needed.
        low = env.action_space.low.astype(np.float32)
        high = env.action_space.high.astype(np.float32)
        raw_actions = rng.uniform(low, high, size=(num_frames, 4)).astype(np.float32) * 0.3

        frames: list[np.ndarray] = []
        states: list[np.ndarray] = []
        controls: list[np.ndarray] = []

        # Record the initial state + frame (before any action).
        frames.append(_obs_to_frame(env.render(), img_size))
        states.append(_obs_to_metric_state(obs))
        controls.append(np.zeros(4, dtype=np.float32))

        hover_thrust = 0.39
        start_pos = obs[9:12].astype(np.float32).copy()
        # Kick schedule: settle → disturb → brake/home.
        kick_at = max(1, num_frames // 4)
        kick_end = min(num_frames - 2, kick_at + max(2, num_frames // 5))
        kick_action = np.array([0.0, 0.75, 0.0, hover_thrust], dtype=np.float32)
        # L-turn targets (meters from start).
        leg1 = start_pos + np.array([0.5, 0.0, 0.0], dtype=np.float32)
        leg2 = start_pos + np.array([0.5, 0.5, 0.0], dtype=np.float32)
        turn_goal = leg1
        turn_reached_leg1 = False

        for t in range(num_frames - 1):
            if control_mode == "hover":
                action = _hover_counter_action(obs, states[0], wind_vec, hover_thrust)
            elif control_mode == "kick":
                if kick_at <= t < kick_end:
                    action = kick_action.copy()
                elif t < kick_at:
                    action = _hover_counter_action(obs, states[0], wind_vec, hover_thrust)
                else:
                    # Brake if still fast, otherwise seek home.
                    lin_vel = obs[6:9].astype(np.float32)
                    if float(np.linalg.norm(lin_vel[:2])) > 0.6:
                        action = _brake_control(obs, hover_thrust)
                    else:
                        action = _seek_control(obs, start_pos, hover_thrust)
            elif control_mode == "turn":
                lin_pos = obs[9:12].astype(np.float32)
                if not turn_reached_leg1 and float(np.linalg.norm(lin_pos - leg1)) < 0.35:
                    turn_reached_leg1 = True
                    turn_goal = leg2
                # Early corner look-ahead once close to leg1.
                if (
                    not turn_reached_leg1
                    and float(np.linalg.norm(lin_pos - leg1)) < 0.55
                ):
                    action = _seek_control(obs, leg2, hover_thrust)
                else:
                    action = _seek_control(obs, turn_goal, hover_thrust)
            else:
                action = raw_actions[t]

            obs, _rew, term, trunc, _info = env.step(action)
            frames.append(_obs_to_frame(env.render(), img_size))
            states.append(_obs_to_metric_state(obs))
            controls.append(action.copy())
            if term or trunc:
                obs, _info = env.reset(seed=seed + t + 1)
                if wind_mps > 0:
                    env.unwrapped.env.register_wind_field_function(
                        lambda time, position, wv=wind_vec: np.broadcast_to(wv, position.shape).copy()
                    )
                frames[-1] = _obs_to_frame(env.render(), img_size)
                states[-1] = _obs_to_metric_state(obs)
                controls[-1] = np.zeros(4, dtype=np.float32)
                start_pos = obs[9:12].astype(np.float32).copy()
                leg1 = start_pos + np.array([0.5, 0.0, 0.0], dtype=np.float32)
                leg2 = start_pos + np.array([0.5, 0.5, 0.0], dtype=np.float32)
                turn_goal = leg1
                turn_reached_leg1 = False
    finally:
        env.close()

    # Truncate or pad to exactly num_frames.
    frames_np = np.stack(frames[:num_frames])
    states_np = np.stack(states[:num_frames])
    controls_np = np.stack(controls[:num_frames])
    if states_np.shape[0] < num_frames:
        pad = num_frames - states_np.shape[0]
        last_frame = frames_np[-1:]
        last_state = states_np[-1:]
        last_ctrl = controls_np[-1:]
        frames_np = np.concatenate([frames_np] + [last_frame] * pad, axis=0)
        states_np = np.concatenate([states_np] + [last_state] * pad, axis=0)
        controls_np = np.concatenate([controls_np] + [last_ctrl] * pad, axis=0)

    actions_np = states_to_actions(states_np)

    return PyFlytClip(
        frames=torch.from_numpy(frames_np),
        actions=torch.from_numpy(actions_np.astype(np.float32)),
        metric_state=torch.from_numpy(states_np.astype(np.float32)),
        control_actions=torch.from_numpy(controls_np.astype(np.float32)),
    )


def _hover_counter_action(
    obs: np.ndarray,
    start_state: np.ndarray,
    wind_vec: np.ndarray,
    hover_thrust: float,
) -> np.ndarray:
    """Altitude PD + lean against XY drift / known wind."""
    lin_vel = obs[6:9].astype(np.float32)
    lin_pos = obs[9:12].astype(np.float32)
    z_des = float(start_state[2])
    kp_v, kp_w = 0.55, 0.18
    vp = float(np.clip(kp_v * lin_vel[1] - kp_w * wind_vec[1], -1.2, 1.2))
    vq = float(np.clip(-kp_v * lin_vel[0] - kp_w * wind_vec[0], -1.2, 1.2))
    thrust = float(
        np.clip(
            hover_thrust + 0.35 * (z_des - float(lin_pos[2])) - 0.15 * float(lin_vel[2]),
            0.0,
            0.8,
        )
    )
    return np.array([vp, vq, 0.0, thrust], dtype=np.float32)


def _brake_control(obs: np.ndarray, hover_thrust: float) -> np.ndarray:
    """Velocity-kill PD (lean against lateral speed)."""
    lin_vel = obs[6:9].astype(np.float32)
    kd_xy, kd_z = 0.9, 0.15
    vp = float(np.clip(kd_xy * lin_vel[1], -1.2, 1.2))
    vq = float(np.clip(-kd_xy * lin_vel[0], -1.2, 1.2))
    thrust = float(np.clip(hover_thrust - kd_z * lin_vel[2], 0.0, 0.8))
    return np.array([vp, vq, 0.0, thrust], dtype=np.float32)


def _seek_control(
    obs: np.ndarray,
    goal_world: np.ndarray,
    hover_thrust: float,
) -> np.ndarray:
    """Reactive PD toward a world-frame goal (same signs as closed_loop)."""
    lin_vel = obs[6:9].astype(np.float32)
    lin_pos = obs[9:12].astype(np.float32)
    err = np.asarray(goal_world, dtype=np.float32) - lin_pos
    dist_xy = float(np.linalg.norm(err[:2]))
    gain = float(np.clip(dist_xy / 0.6, 0.25, 1.0))
    kp_xy, kd_xy = 1.0, 0.55
    vp = float(np.clip(-(gain * kp_xy * err[1] - kd_xy * lin_vel[1]), -1.2, 1.2))
    vq = float(np.clip(gain * kp_xy * err[0] - kd_xy * lin_vel[0], -1.2, 1.2))
    thrust = float(
        np.clip(hover_thrust + 0.35 * err[2] - 0.15 * lin_vel[2], 0.0, 0.8)
    )
    return np.array([vp, vq, 0.0, thrust], dtype=np.float32)


def _obs_to_frame(frame: np.ndarray, img_size: int) -> np.ndarray:
    """Convert PyFlyt RGBA render (H,W,4) uint8 to (3,H,W) float32 in [0,1]."""
    if frame.ndim == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]  # drop alpha
    if frame.shape[0] != img_size or frame.shape[1] != img_size:
        # PyFlyt may return a different resolution; resize via simple crop/scale.
        frame = frame[:img_size, :img_size]
    # (H, W, 3) uint8 -> (3, H, W) float32 in [0, 1]
    out = np.ascontiguousarray(frame).astype(np.float32) / 255.0
    return np.transpose(out, (2, 0, 1))


def _obs_to_metric_state(obs: np.ndarray) -> np.ndarray:
    """Extract a 12-D metric state from the PyFlyt gym observation.

    Obs layout (euler): ``[ang_vel(3), ang_pos(3) rpy rad, lin_vel(3), lin_pos(3), ...]``.
    PyFlyt docs: ``ang_vel`` and ``lin_vel`` are **body-frame**; ``ang_pos`` /
    ``lin_pos`` are ground/inertial. ``ang_vel`` is in **rad/s**.

    Output layout (matches :class:`ControlIntegrator`):
      ``[pos_world(3), vel_world(3), euler_att_deg(yaw,pitch,roll), ang_vel_body_pqr_deg(3)]``
    """
    ang_vel_body_rad = obs[0:3].astype(np.float64)  # (p, q, r) rad/s
    ang_pos_rpy_rad = obs[3:6]
    lin_vel_body = obs[6:9].astype(np.float64)
    lin_pos = obs[9:12].astype(np.float32)
    att_deg_ypr = _euler_rad_to_deg_yaw_pitch_roll(ang_pos_rpy_rad)
    R = _euler_ypr_deg_to_R(att_deg_ypr)  # body → world
    vel_world = (R @ lin_vel_body).astype(np.float32)
    ang_vel_deg = np.degrees(ang_vel_body_rad).astype(np.float32)  # (p, q, r) deg/s
    return np.concatenate([lin_pos, vel_world, att_deg_ypr, ang_vel_deg], axis=0).astype(
        np.float32
    )


class PyFlytClipsDataset(Dataset):
    """A reproducible dataset of PyFlyt-generated drone clips.

    Each index maps deterministically to one clip via a per-sample seed, so the
    dataset is fully reproducible (mirrors ``SyntheticDroneClips``).

    Stress mix (fractions are applied in order, remainder is random):

    * ``wind_fraction`` — hover / wind-counter under constant wind
    * ``kick_fraction`` — disturb then brake/home (recover supervision)
    * ``turn_fraction`` — L-path seek (corner supervision)

    **Leak warning:** ``hover`` / ``kick`` / ``turn`` build controls from the
    current observation (state-dependent PD). Do **not** use nonzero
    ``wind/kick/turn_fraction`` for leak-free headline ablations — keep the
    default fractions at 0 (random exogenous RNG controls only).
    """

    def __init__(
        self,
        num_clips: int = 256,
        num_frames: int = 8,
        img_size: int = 64,
        seed: int = 0,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 10.0,
        agent_hz: int = 40,
        wind_mps: float = 0.0,
        wind_fraction: float = 0.0,
        wind_mps_max: float | None = None,
        kick_fraction: float = 0.0,
        turn_fraction: float = 0.0,
    ) -> None:
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.img_size = img_size
        self.seed = seed
        self.flight_dome_size = flight_dome_size
        self.max_duration_seconds = max_duration_seconds
        self.agent_hz = agent_hz
        self.wind_mps = float(wind_mps)
        self.wind_fraction = float(np.clip(wind_fraction, 0.0, 1.0))
        self.wind_mps_max = float(wind_mps_max) if wind_mps_max is not None else max(self.wind_mps, 3.0)
        self.kick_fraction = float(np.clip(kick_fraction, 0.0, 1.0))
        self.turn_fraction = float(np.clip(turn_fraction, 0.0, 1.0))
        total = self.wind_fraction + self.kick_fraction + self.turn_fraction
        if total > 1.0 + 1e-6:
            raise ValueError(
                f"wind+kick+turn fractions must sum to ≤1, got {total:.3f}"
            )
        if total > 0.0:
            warnings.warn(
                "PyFlytClipsDataset: nonzero wind/kick/turn_fraction uses "
                "state-dependent PD controls (leaky for leak-free claims). "
                "Default published recipes keep all fractions at 0.",
                UserWarning,
                stacklevel=2,
            )

    def __len__(self) -> int:
        return self.num_clips

    def _mode_for_index(self, idx: int) -> str:
        """Deterministic stress assignment from index (percent buckets)."""
        bucket = idx % 100
        wind_cut = int(100 * self.wind_fraction)
        kick_cut = wind_cut + int(100 * self.kick_fraction)
        turn_cut = kick_cut + int(100 * self.turn_fraction)
        if bucket < wind_cut:
            return "wind"
        if bucket < kick_cut:
            return "kick"
        if bucket < turn_cut:
            return "turn"
        return "random"

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        clip_seed = self.seed * 100_003 + idx
        mode = self._mode_for_index(idx)
        if mode == "wind":
            rng = np.random.default_rng(clip_seed + 17)
            w = float(rng.uniform(max(self.wind_mps, 0.5), self.wind_mps_max))
            angle = float(rng.uniform(0, 2 * np.pi))
            direction = (np.cos(angle), np.sin(angle), 0.0)
            control_mode = "hover"
        elif mode == "kick":
            w = 0.0
            direction = (1.0, 0.0, 0.0)
            control_mode = "kick"
        elif mode == "turn":
            w = 0.0
            direction = (1.0, 0.0, 0.0)
            control_mode = "turn"
        else:
            w = 0.0
            direction = (1.0, 0.0, 0.0)
            control_mode = "random"
        clip = generate_clip(
            seed=clip_seed,
            num_frames=self.num_frames,
            img_size=self.img_size,
            flight_dome_size=self.flight_dome_size,
            max_duration_seconds=self.max_duration_seconds,
            agent_hz=self.agent_hz,
            wind_mps=w,
            wind_direction=direction,
            control_mode=control_mode,
        )
        return clip.frames, clip.actions, clip.metric_state, clip.control_actions


def build_pyflyt_dataloaders(
    batch_size: int = 16,
    num_frames: int = 8,
    img_size: int = 64,
    num_train: int = 256,
    num_val: int = 32,
    num_workers: int = 0,
    seed: int = 0,
    wind_mps: float = 0.0,
    wind_fraction: float = 0.0,
    wind_mps_max: float | None = None,
    kick_fraction: float = 0.0,
    turn_fraction: float = 0.0,
    **env_kwargs: Any,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Train/val loaders over disjoint PyFlyt clip seeds.

    Note: PyFlyt is not fork-safe by default, so ``num_workers`` should be 0.
    Pre-generating clips to disk (via ``scripts/generate_pyflyt_cache.py``) is
    recommended for larger datasets and is multiprocessing-safe.
    """
    common = dict(
        num_frames=num_frames,
        img_size=img_size,
        wind_mps=wind_mps,
        wind_fraction=wind_fraction,
        wind_mps_max=wind_mps_max,
        kick_fraction=kick_fraction,
        turn_fraction=turn_fraction,
        **env_kwargs,
    )
    train = PyFlytClipsDataset(num_clips=num_train, seed=seed, **common)
    val = PyFlytClipsDataset(num_clips=num_val, seed=seed + 9973, **common)
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False,
    )
    return train_loader, val_loader
