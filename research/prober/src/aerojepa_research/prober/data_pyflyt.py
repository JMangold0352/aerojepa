"""PyFlyt data generator for AeroProber.

Generates clips of ``(frames, actions, metric_state)`` where:
- ``frames``    (T, 3, H, W) float32 in [0, 1] -- rendered RGB from the drone cam.
- ``actions``   (T, 6) float32 -- AeroJEPA ACTION_COLUMNS convention
                 (dx, dy, d_altitude, d_yaw, d_pitch, d_roll), derived from
                 consecutive PyFlyt states so the frozen action-conditioned
                 predictor (if any) sees in-distribution inputs.
- ``metric_state`` (T, 12) float32 -- [pos(3), vel(3), euler_att_deg(3), ang_vel(3)]
                    ground truth for the supervised prober loss. Attitude is
                    in DEGREES (yaw, pitch, roll order) and wrapped to (-180, 180].

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
    states : (T, 12)  [pos, vel, euler_att_deg(yaw,pitch,roll), ang_vel] per frame

    Returns
    -------
    actions : (T, 6)  (dx, dy, d_altitude, d_yaw, d_pitch, d_roll)
              The first row's angular channels are zero (no previous frame);
              the linear channels carry the first frame's velocity (matching
              ``telemetry.derive_actions_from_raw``, which copies vgx/vgy/vgz
              for every row including the first).
    """
    T = states.shape[0]
    actions = np.zeros((T, ACTION_DIM), dtype=np.float32)
    if T < 2:
        return actions
    vel = states[:, 3:6]           # velocity (PyFlyt: world-frame; treated as body proxy)
    att = states[:, 6:9]           # yaw, pitch, roll (deg)
    # Linear channels: the velocity itself (matches telemetry: actions[:,0]=vgx, etc.).
    actions[:, 0:3] = vel
    # Angular channels: wrapped frame-to-frame attitude deltas.
    actions[1:, 3] = _wrap_degrees(np.diff(att[:, 0]))  # d_yaw
    actions[1:, 4] = np.diff(att[:, 1])                  # d_pitch
    actions[1:, 5] = np.diff(att[:, 2])                  # d_roll
    return actions


@dataclass
class PyFlytClip:
    """One generated clip."""

    frames: torch.Tensor       # (T, 3, H, W) float32 in [0, 1]
    actions: torch.Tensor      # (T, 6) float32
    metric_state: torch.Tensor # (T, 12) float32


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
) -> PyFlytClip:
    """Render one deterministic PyFlyt clip with full metric state.

    The drone is driven by random control actions (angular rates + thrust) from
    a seeded RNG, producing physically-plausible motion. We record rendered
    frames + the full metric state at each control step, then derive AeroJEPA-
    convention actions from consecutive states.

    Parameters
    ----------
    seed : int
        Reproducibility seed for both the Python RNG and the gym env.
    num_frames : int
        Clip length in frames.
    img_size : int
        Rendered frame size (matches AeroJEPA img_size; 64 for synth configs).
    flight_dome_size, max_duration_seconds, agent_hz : gym env params.

    Returns
    -------
    PyFlytClip
    """
    import gymnasium
    import PyFlyt.gym_envs  # noqa: F401  -- registers envs

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

        # Pre-sample the entire control-action sequence from our own seeded RNG
        # so clips are reproducible. The action space is (vp, vq, vr, T) with
        # bounds [-pi, pi]^3 x [0, 0.8]; we sample uniformly and scale down for
        # gentler motion that stays in-dome.
        low = env.action_space.low.astype(np.float32)
        high = env.action_space.high.astype(np.float32)
        raw_actions = rng.uniform(low, high, size=(num_frames, 4)).astype(np.float32) * 0.3

        frames: list[np.ndarray] = []
        states: list[np.ndarray] = []

        # Record the initial state + frame (before any action).
        frames.append(_obs_to_frame(env.render(), img_size))
        states.append(_obs_to_metric_state(obs))

        # Step the env with our pre-sampled control actions.
        for t in range(num_frames - 1):
            obs, _rew, term, trunc, _info = env.step(raw_actions[t])
            frames.append(_obs_to_frame(env.render(), img_size))
            states.append(_obs_to_metric_state(obs))
            if term or trunc:
                # Reset and continue -- the clip will be padded if needed.
                obs, _info = env.reset(seed=seed + t + 1)
                frames[-1] = _obs_to_frame(env.render(), img_size)
                states[-1] = _obs_to_metric_state(obs)
    finally:
        env.close()

    # Truncate or pad to exactly num_frames.
    frames_np = np.stack(frames[:num_frames])
    states_np = np.stack(states[:num_frames])
    if states_np.shape[0] < num_frames:
        pad = num_frames - states_np.shape[0]
        last_frame = frames_np[-1:]
        last_state = states_np[-1:]
        frames_np = np.concatenate([frames_np] + [last_frame] * pad, axis=0)
        states_np = np.concatenate([states_np] + [last_state] * pad, axis=0)

    actions_np = states_to_actions(states_np)

    return PyFlytClip(
        frames=torch.from_numpy(frames_np),
        actions=torch.from_numpy(actions_np.astype(np.float32)),
        metric_state=torch.from_numpy(states_np.astype(np.float32)),
    )


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

    Obs layout (euler): [ang_vel(3), ang_pos(3) rpy rad, lin_vel(3), lin_pos(3), action(4), aux(4)].
    Output: [pos(3), vel(3), euler_att_deg(3) as (yaw,pitch,roll), ang_vel(3)].
    """
    ang_vel = obs[0:3]
    ang_pos_rpy_rad = obs[3:6]
    lin_vel = obs[6:9]
    lin_pos = obs[9:12]
    att_deg_ypr = _euler_rad_to_deg_yaw_pitch_roll(ang_pos_rpy_rad)
    return np.concatenate([lin_pos, lin_vel, att_deg_ypr, ang_vel], axis=0).astype(np.float32)


class PyFlytClipsDataset(Dataset):
    """A reproducible dataset of PyFlyt-generated drone clips.

    Each index maps deterministically to one clip via a per-sample seed, so the
    dataset is fully reproducible (mirrors ``SyntheticDroneClips``).
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
    ) -> None:
        self.num_clips = num_clips
        self.num_frames = num_frames
        self.img_size = img_size
        self.seed = seed
        self.flight_dome_size = flight_dome_size
        self.max_duration_seconds = max_duration_seconds
        self.agent_hz = agent_hz

    def __len__(self) -> int:
        return self.num_clips

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        clip = generate_clip(
            seed=self.seed * 100_003 + idx,
            num_frames=self.num_frames,
            img_size=self.img_size,
            flight_dome_size=self.flight_dome_size,
            max_duration_seconds=self.max_duration_seconds,
            agent_hz=self.agent_hz,
        )
        return clip.frames, clip.actions, clip.metric_state


def build_pyflyt_dataloaders(
    batch_size: int = 16,
    num_frames: int = 8,
    img_size: int = 64,
    num_train: int = 256,
    num_val: int = 32,
    num_workers: int = 0,
    seed: int = 0,
    **env_kwargs: Any,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Train/val loaders over disjoint PyFlyt clip seeds.

    Note: PyFlyt is not fork-safe by default, so ``num_workers`` should be 0.
    Pre-generating clips to disk (via ``scripts/generate_pyflyt_cache.py``) is
    recommended for larger datasets and is multiprocessing-safe.
    """
    train = PyFlytClipsDataset(
        num_clips=num_train, num_frames=num_frames, img_size=img_size,
        seed=seed, **env_kwargs,
    )
    val = PyFlytClipsDataset(
        num_clips=num_val, num_frames=num_frames, img_size=img_size,
        seed=seed + 9973, **env_kwargs,
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False,
    )
    return train_loader, val_loader
