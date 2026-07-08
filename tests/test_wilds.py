from __future__ import annotations

import json

import numpy as np
import pytest

from aerojepa.data.telemetry import ACTION_COLUMNS
from aerojepa.data.wilds import (
    load_parrot_log,
    parrot_log_to_actions,
    resample_log_to_frames,
)


@pytest.fixture()
def sample_parrot_json(tmp_path):
    headers = [
        "time", "battery_level", "controller_gps_latitude", "controller_gps_longitude",
        "flying_state", "alert_state", "wifi_signal", "product_gps_available",
        "product_gps_longitude", "product_gps_latitude", "product_gps_position_error",
        "product_gps_sv_number", "speed_vx", "speed_vy", "speed_vz",
        "angle_phi", "angle_theta", "angle_psi", "altitude", "flip_type", "speed",
    ]
    rows = []
    for i in range(20):
        rows.append([
            i * 100, 90, 0, 0, 7, 0, -30, True, 0, 0, 0, 10,
            1.0, -0.5, 0.2,
            0.1 * i, 0.05 * i, 0.2 * i,
            10.0 + 0.1 * i, 0, 1.0,
        ])
    path = tmp_path / "flight.json"
    path.write_text(json.dumps({"details_headers": headers, "details_data": rows}))
    return path


def test_load_parrot_log_shape(sample_parrot_json) -> None:
    log = load_parrot_log(sample_parrot_json)
    assert log.shape == (20, 8)
    assert log[1, 0] == pytest.approx(0.1)  # 100 ms -> 0.1 s


def test_resample_log_to_frames(sample_parrot_json) -> None:
    log = load_parrot_log(sample_parrot_json)
    out = resample_log_to_frames(log, num_frames=10, fps=10.0)
    assert out.shape == (10, 8)


def test_parrot_log_to_actions(sample_parrot_json) -> None:
    log = load_parrot_log(sample_parrot_json)
    actions = parrot_log_to_actions(log, num_frames=10, fps=10.0)
    assert actions.shape == (10, len(ACTION_COLUMNS))
    assert np.allclose(actions[0, 3:6], 0.0)  # first angular deltas zero
