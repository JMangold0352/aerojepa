"""Test the extended Wilds converter that preserves absolute metric state."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aerojepa_research.prober.wilds_state import (
    STATE_COLUMNS,
    convert_wilds_with_state,
    parrot_log_to_state,
)


pytestmark = pytest.mark.slow


def test_state_columns_layout():
    assert len(STATE_COLUMNS) == 12
    # pos(3), vel(3), att(3), ang_vel(3)
    assert STATE_COLUMNS[:3] == ("pos_x", "pos_y", "pos_z")
    assert STATE_COLUMNS[3:6] == ("vel_x", "vel_y", "vel_z")
    assert STATE_COLUMNS[6:9] == ("yaw", "pitch", "roll")
    assert STATE_COLUMNS[9:12] == ("av_yaw", "av_pitch", "av_roll")


def test_parrot_log_to_state_shapes_and_ranges():
    # Synthetic log: [t, vx, vy, vz, altitude, yaw, pitch, roll] (angles rad).
    N = 20
    t = np.arange(N, dtype=np.float32) / 15.0
    log = np.zeros((N, 8), dtype=np.float32)
    log[:, 0] = t
    log[:, 1] = 1.0  # constant vx
    log[:, 4] = 2.0  # altitude
    log[:, 5] = np.linspace(0, 1.0, N)  # yaw 0 -> 1 rad

    state = parrot_log_to_state(log, N, fps=15.0)
    assert state.shape == (N, 12)
    # Attitude in degrees, wrapped to (-180, 180].
    att = state[:, 6:9]
    assert (att >= -180.0).all() and (att < 180.0).all()
    # pos_z = altitude.
    assert np.allclose(state[:, 2], 2.0)
    # Constant vx -> pos_x grows roughly linearly after frame 0.
    assert state[5, 0] > state[1, 0] > 0.0
    # First-frame angular velocities are zero.
    assert np.allclose(state[0, 9:12], 0.0)


def test_convert_wilds_with_state_on_real_data():
    """End-to-end on the real raw Wilds tree (skipped if absent)."""
    raw = Path("data/raw/thewilds")
    if not raw.exists():
        pytest.skip("raw Wilds data not present")
    out = Path("data/flights_with_state")
    out.mkdir(exist_ok=True)
    written = convert_wilds_with_state(raw, out, link_videos=True)
    assert len(written) > 0
    # Each video should have both an actions CSV and a state CSV.
    for v in written[:3]:
        actions_csv = v.with_suffix(".csv")
        state_csv = v.parent / (v.stem + "_state.csv")
        assert actions_csv.exists(), f"missing {actions_csv}"
        assert state_csv.exists(), f"missing {state_csv}"
        # State CSV has 12 columns + header.
        rows = np.loadtxt(state_csv, delimiter=",", skiprows=1, ndmin=2)
        assert rows.shape[1] == 12
        # Attitude wrapped.
        att = rows[:, 6:9]
        assert (att >= -180.0).all() and (att < 180.0).all()
