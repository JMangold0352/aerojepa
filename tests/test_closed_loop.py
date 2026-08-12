from __future__ import annotations

import numpy as np
import torch

from aerojepa.sim.closed_loop import aerojepa_to_pyflyt, classify_failure_mode


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
