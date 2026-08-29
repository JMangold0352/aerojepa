from __future__ import annotations

import numpy as np
import torch

from aerojepa.sim.closed_loop import (
    aerojepa_to_pyflyt,
    classify_failure_mode,
    run_closed_loop_episode,
)
from aerojepa.sim.vehicle import (
    RATE_LIMIT,
    THRUST_MAX,
    VehicleObs,
    VehicleState,
    clip_control,
)


def test_aerojepa_to_pyflyt_shape_and_bounds() -> None:
    out = aerojepa_to_pyflyt(np.zeros(6, dtype=np.float32), hover_thrust=0.39)
    assert out.shape == (4,)
    assert out.dtype == np.float32
    assert abs(float(out[3]) - 0.39) < 1e-5
    assert float(out[0]) == 0.0 and float(out[1]) == 0.0 and float(out[2]) == 0.0


def test_aerojepa_to_pyflyt_maps_lateral_deltas() -> None:
    """+dx → +vq, +dy → -vp (QuadX-Hover-v4 empirical convention)."""
    plus_x = aerojepa_to_pyflyt(
        torch.tensor([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
        xy_scale=3.5,
        rate_scale=0.0,
    )
    plus_y = aerojepa_to_pyflyt(
        torch.tensor([0.0, 0.2, 0.0, 0.0, 0.0, 0.0]),
        xy_scale=3.5,
        rate_scale=0.0,
    )
    assert float(plus_x[1]) > 0.0  # vq
    assert float(plus_y[0]) < 0.0  # vp
    assert abs(float(plus_x[0])) < 1e-6
    assert abs(float(plus_y[1])) < 1e-6

    hot = aerojepa_to_pyflyt(
        torch.tensor([0.1, -0.1, 0.5, 1.0, -1.0, 2.0]),
        hover_thrust=0.39,
        rate_scale=10.0,
        alt_scale=1.0,
    )
    assert abs(float(hot[0])) <= np.pi + 1e-4
    assert abs(float(hot[1])) <= np.pi + 1e-4
    assert abs(float(hot[2])) <= np.pi + 1e-4
    assert -1e-5 <= float(hot[3]) <= 0.8 + 1e-5


def test_classify_failure_wind_ok_vs_excessive_drift() -> None:
    mode, detail = classify_failure_mode(
        task="wind_gust",
        survived=True,
        terminated=False,
        truncated=False,
        hit_max_steps=True,
        final_altitude=1.0,
        max_xy_drift=0.4,
        reached=None,
        recovered=None,
        wind_mps=2.0,
        wind_drift_fail=1.5,
    )
    assert mode == "ok"
    assert "2.0" in detail

    mode, _ = classify_failure_mode(
        task="wind_gust",
        survived=True,
        terminated=False,
        truncated=False,
        hit_max_steps=True,
        final_altitude=1.0,
        max_xy_drift=2.5,
        reached=None,
        recovered=None,
        wind_mps=2.0,
        wind_drift_fail=1.5,
    )
    assert mode == "excessive_drift"


def test_classify_failure_aggressive_turn_legs() -> None:
    mode, detail = classify_failure_mode(
        task="aggressive_turn",
        survived=True,
        terminated=False,
        truncated=False,
        hit_max_steps=True,
        final_altitude=1.0,
        max_xy_drift=1.0,
        reached=False,
        recovered=None,
        waypoints_reached=1,
        waypoints_total=2,
    )
    assert mode == "missed_turn"
    assert "1/2" in detail

    mode, _ = classify_failure_mode(
        task="aggressive_turn",
        survived=True,
        terminated=False,
        truncated=False,
        hit_max_steps=False,
        final_altitude=1.0,
        max_xy_drift=1.1,
        reached=True,
        recovered=None,
        waypoints_reached=2,
        waypoints_total=2,
    )
    assert mode == "ok"


def test_classify_failure_crash_beats_other_labels() -> None:
    mode, _ = classify_failure_mode(
        task="wind_gust",
        survived=False,
        terminated=True,
        truncated=False,
        hit_max_steps=False,
        final_altitude=0.05,
        max_xy_drift=3.0,
        reached=None,
        recovered=None,
        wind_mps=5.0,
    )
    assert mode == "crash"


def test_clip_control_bounds() -> None:
    hot = clip_control(np.array([10.0, -10.0, 4.0, 2.0], dtype=np.float32))
    assert abs(float(hot[0])) <= RATE_LIMIT + 1e-6
    assert abs(float(hot[1])) <= RATE_LIMIT + 1e-6
    assert abs(float(hot[2])) <= RATE_LIMIT + 1e-6
    assert 0.0 <= float(hot[3]) <= THRUST_MAX + 1e-6


class _RecordingVehicle:
    """Fake Vehicle with no PyFlyt; records step controls."""

    def __init__(self, img_size: int = 64) -> None:
        self.img_size = img_size
        self.controls: list[np.ndarray] = []
        self._t = 0
        self._last: VehicleObs | None = None

    def _make_obs(self, *, reward: float = 0.0) -> VehicleObs:
        rgb = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
        state = VehicleState(
            xyz=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            vxyz=np.zeros(3, dtype=np.float32),
            timestamp_s=float(self._t) / 40.0,
        )
        return VehicleObs(rgb=rgb, state=state, reward=reward)

    def reset(self, seed: int | None = None) -> VehicleObs:
        del seed
        self._t = 0
        self.controls.clear()
        self._last = self._make_obs()
        return self._last

    def rgb(self) -> np.ndarray:
        assert self._last is not None
        return self._last.rgb

    def state(self) -> VehicleState:
        assert self._last is not None
        return self._last.state

    def step(self, control: np.ndarray) -> VehicleObs:
        self.controls.append(clip_control(control).copy())
        self._t += 1
        self._last = self._make_obs(reward=0.0)
        return self._last

    def close(self) -> None:
        return None


def test_fake_vehicle_inert_episode_records_controls() -> None:
    vehicle = _RecordingVehicle(img_size=32)
    ep = run_closed_loop_episode(
        model=None,
        device=torch.device("cpu"),
        policy="inert",
        task="hover",
        img_size=32,
        max_steps=5,
        seed=0,
        record_frames=False,
        assist_altitude=False,
        vehicle=vehicle,
    )
    assert ep.steps == 5
    assert len(vehicle.controls) == 5
    assert all(c.shape == (4,) for c in vehicle.controls)
    assert all(float(np.linalg.norm(c)) < 1e-6 for c in vehicle.controls)


def test_fake_vehicle_hover_episode_records_thrust() -> None:
    vehicle = _RecordingVehicle(img_size=32)
    ep = run_closed_loop_episode(
        model=None,
        device=torch.device("cpu"),
        policy="hover",
        task="hover",
        img_size=32,
        max_steps=3,
        seed=1,
        record_frames=False,
        assist_altitude=False,
        hover_thrust=0.39,
        vehicle=vehicle,
    )
    assert ep.steps == 3
    assert len(vehicle.controls) == 3
    assert abs(float(vehicle.controls[0][3]) - 0.39) < 1e-5
    assert ep.watchdog_holds == 0
    assert ep.budget_ms == 25.0
    assert ep.mean_loop_ms is not None
    assert len(ep.loop_ms) == 3


def test_watchdog_hold_on_nan_control() -> None:
    vehicle = _RecordingVehicle(img_size=32)
    ep = run_closed_loop_episode(
        model=None,
        device=torch.device("cpu"),
        policy="hover",
        task="hover",
        img_size=32,
        max_steps=4,
        seed=2,
        record_frames=False,
        assist_altitude=False,
        hover_thrust=0.39,
        vehicle=vehicle,
        debug_inject_nan_control=True,
    )
    assert ep.watchdog_holds == 4
    assert len(vehicle.controls) == 4
    for c in vehicle.controls:
        assert np.isfinite(c).all()
        assert abs(float(c[0])) < 1e-6
        assert abs(float(c[1])) < 1e-6
        assert abs(float(c[2])) < 1e-6
        assert abs(float(c[3]) - 0.39) < 1e-5


def test_watchdog_hold_on_slow_plan() -> None:
    vehicle = _RecordingVehicle(img_size=32)
    # budget=25 ms at 40 Hz; 2× = 50 ms. Sleep past that during encode/plan.
    ep = run_closed_loop_episode(
        model=None,
        device=torch.device("cpu"),
        policy="hover",
        task="hover",
        img_size=32,
        max_steps=2,
        seed=3,
        record_frames=False,
        assist_altitude=False,
        hover_thrust=0.39,
        agent_hz=40,
        vehicle=vehicle,
        debug_plan_delay_ms=60.0,
    )
    assert ep.watchdog_holds >= 1
    assert all(np.isfinite(c).all() for c in vehicle.controls)
