"""Vehicle protocol for closed-loop research demos.

The episode loop talks only to :class:`Vehicle` (reset / rgb / state / step /
close). :class:`PyFlytVehicle` is the current adapter; a later Tello can
implement the same surface. This is **not** a flight controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from aerojepa.sim.simulators import make_pyflyt_env

DEFAULT_ENV_ID = "PyFlyt/QuadX-Hover-v4"
DEFAULT_HOVER_THRUST = 0.39
RATE_LIMIT = float(np.pi)
THRUST_MAX = 0.8


@dataclass
class VehicleState:
    """Minimal vehicle state for planning / assist PD."""

    xyz: np.ndarray  # (3,) world position
    vxyz: np.ndarray  # (3,) world linear velocity
    euler_rad: np.ndarray | None = None  # (3,) roll,pitch,yaw if known
    rates_rad: np.ndarray | None = None  # (3,) body p,q,r if known
    timestamp_s: float = 0.0


@dataclass
class VehicleObs:
    """One observation from :meth:`Vehicle.step` / :meth:`Vehicle.reset`."""

    rgb: np.ndarray  # (3, H, W) float32 in [0, 1]
    state: VehicleState
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False


def clip_control(control: np.ndarray) -> np.ndarray:
    """Clip ``(vp, vq, vr, T)`` to rate ±π and thrust ``[0, 0.8]``."""
    c = np.asarray(control, dtype=np.float32).reshape(-1)
    if c.shape[0] != 4:
        raise ValueError(f"expected 4-D control, got shape {c.shape}")
    return np.array(
        [
            float(np.clip(c[0], -RATE_LIMIT, RATE_LIMIT)),
            float(np.clip(c[1], -RATE_LIMIT, RATE_LIMIT)),
            float(np.clip(c[2], -RATE_LIMIT, RATE_LIMIT)),
            float(np.clip(c[3], 0.0, THRUST_MAX)),
        ],
        dtype=np.float32,
    )


@runtime_checkable
class Vehicle(Protocol):
    """Small vehicle surface shared by PyFlyt (now) and optional later hardware."""

    def reset(self, seed: int | None = None) -> VehicleObs: ...

    def rgb(self) -> np.ndarray:
        """``(3, H, W)`` float32 in ``[0, 1]``, model input size."""
        ...

    def state(self) -> VehicleState: ...

    def step(self, control: np.ndarray) -> VehicleObs:
        """Apply clipped ``(vp, vq, vr, T)`` control; return rgb + state."""
        ...

    def close(self) -> None: ...


def _obs_xyz_vxyz(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract linear position / velocity from PyFlyt euler observation."""
    o = np.asarray(obs, dtype=np.float32).reshape(-1)
    lin_vel = o[6:9].copy()
    lin_pos = o[9:12].copy()
    return lin_pos, lin_vel


def _state_from_pyflyt_obs(obs: np.ndarray, *, timestamp_s: float) -> VehicleState:
    o = np.asarray(obs, dtype=np.float32).reshape(-1)
    xyz, vxyz = _obs_xyz_vxyz(o)
    rates = o[0:3].astype(np.float32).copy() if o.shape[0] >= 3 else None
    euler = o[3:6].astype(np.float32).copy() if o.shape[0] >= 6 else None
    return VehicleState(
        xyz=xyz,
        vxyz=vxyz,
        euler_rad=euler,
        rates_rad=rates,
        timestamp_s=float(timestamp_s),
    )


def _render_frame(env: Any, img_size: int) -> np.ndarray:
    """PyFlyt RGBA render → ``(3, H, W)`` float32 in ``[0, 1]``."""
    frame = env.render()
    if frame is None:
        raise RuntimeError(
            "env.render() returned None; create the env with render_mode='rgb_array'"
        )
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


def make_constant_wind_fn(
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


class PyFlytVehicle:
    """PyFlyt QuadX adapter. Gym / aviary details stay inside this class."""

    def __init__(
        self,
        *,
        img_size: int = 64,
        env_id: str = DEFAULT_ENV_ID,
        agent_hz: int = 40,
        flight_dome_size: float = 5.0,
        max_duration_seconds: float = 12.0,
    ) -> None:
        self.img_size = int(img_size)
        self.agent_hz = int(agent_hz)
        self._env = make_pyflyt_env(
            env_id,
            render_mode="rgb_array",
            angle_representation="euler",
            render_resolution=(self.img_size, self.img_size),
            flight_dome_size=flight_dome_size,
            max_duration_seconds=max_duration_seconds,
            agent_hz=self.agent_hz,
        )
        self._step_i = 0
        self._last: VehicleObs | None = None

    def reset(self, seed: int | None = None) -> VehicleObs:
        raw, _info = self._env.reset(seed=seed)
        self._step_i = 0
        self._last = self._pack(raw, reward=0.0, terminated=False, truncated=False)
        return self._last

    def rgb(self) -> np.ndarray:
        if self._last is None:
            raise RuntimeError("call reset() before rgb()")
        return self._last.rgb

    def state(self) -> VehicleState:
        if self._last is None:
            raise RuntimeError("call reset() before state()")
        return self._last.state

    def step(self, control: np.ndarray) -> VehicleObs:
        action = clip_control(control)
        raw, reward, terminated, truncated, _info = self._env.step(action)
        self._step_i += 1
        self._last = self._pack(
            raw,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
        )
        return self._last

    def close(self) -> None:
        self._env.close()

    def register_wind_field_function(self, wind_fn: Any) -> None:
        """Attach a wind field to the live aviary (after ``reset``)."""
        aviary = getattr(getattr(self._env, "unwrapped", self._env), "env", None)
        if aviary is None or not hasattr(aviary, "register_wind_field_function"):
            raise RuntimeError(
                "PyFlyt env has no register_wind_field_function; cannot run wind_gust."
            )
        aviary.register_wind_field_function(wind_fn)

    def _pack(
        self,
        raw_obs: np.ndarray,
        *,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> VehicleObs:
        ts = float(self._step_i) / float(self.agent_hz)
        return VehicleObs(
            rgb=_render_frame(self._env, self.img_size),
            state=_state_from_pyflyt_obs(raw_obs, timestamp_s=ts),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
        )
