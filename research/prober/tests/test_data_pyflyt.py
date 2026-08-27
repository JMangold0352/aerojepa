"""End-to-end smoke test for the PyFlyt data generator.

Marked as integration (slow) because it spins up PyBullet. Run with:
    pytest research/prober/tests/test_data_pyflyt.py -v -s

NOTE: must run OUTSIDE the Cursor sandbox (PyFlyt segfaults inside it).
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


def test_generate_clip_shapes():
    from aerojepa_research.prober.data_pyflyt import generate_clip
    import torch

    clip = generate_clip(seed=0, num_frames=8, img_size=64)
    assert clip.frames.shape == (8, 3, 64, 64)
    assert clip.frames.dtype.is_floating_point
    assert 0.0 <= clip.frames.min() <= 1.0 + 1e-6
    assert clip.frames.max() <= 1.0 + 1e-6

    assert clip.actions.shape == (8, 6)
    assert clip.actions.dtype.is_floating_point

    assert clip.metric_state.shape == (8, 12)
    # Attitude in degrees must be in [-180, 180).
    att = clip.metric_state[:, 6:9]
    assert (att >= -180.0).all() and (att < 180.0).all()

    # Control actions: raw (vp, vq, vr, T) -- genuinely exogenous commands.
    assert clip.control_actions.shape == (8, 4)
    assert clip.control_actions.dtype.is_floating_point
    # First frame's control is zero by convention (no preceding step).
    assert torch.allclose(clip.control_actions[0], torch.zeros(4))


def test_generate_clip_reproducible():
    from aerojepa_research.prober.data_pyflyt import generate_clip

    c1 = generate_clip(seed=42, num_frames=8, img_size=64)
    c2 = generate_clip(seed=42, num_frames=8, img_size=64)
    assert c1.frames.equal(c2.frames)
    assert c1.actions.equal(c2.actions)
    assert c1.metric_state.equal(c2.metric_state)
    assert c1.control_actions.equal(c2.control_actions)


def test_states_to_actions_matches_telemetry_convention():
    """Linear channels must be body velocity (R^T v_world); angular = wrapped Δatt."""
    from aerojepa_research.prober.data_pyflyt import states_to_actions, _euler_ypr_deg_to_R
    import numpy as np

    states = np.zeros((5, 12), dtype=np.float32)
    states[:, 3:6] = np.arange(5).reshape(-1, 1) * 0.5  # world vel
    states[:, 6] = [10.0, 25.0, 45.0, 80.0, 120.0]  # yaw (deg)

    actions = states_to_actions(states)
    assert actions.shape == (5, 6)
    for t in range(5):
        R = _euler_ypr_deg_to_R(states[t, 6:9])
        expect = R.T @ states[t, 3:6]
        np.testing.assert_allclose(actions[t, 0:3], expect, atol=1e-5)
    assert np.allclose(actions[0, 3:6], 0.0)
    assert abs(actions[1, 3] - 15.0) < 1e-5
    assert abs(actions[2, 3] - 20.0) < 1e-5


def test_obs_to_metric_state_units_and_frames():
    """PyFlyt body rad/s → world vel + body deg/s; attitude reordered to YPR deg."""
    from aerojepa_research.prober.data_pyflyt import _obs_to_metric_state
    import numpy as np

    # Level attitude (rpy=0), body vel [1,0,0], body rate [0,0,π] rad/s → 180 deg/s yaw rate
    obs = np.zeros(20, dtype=np.float64)
    obs[0:3] = [0.0, 0.0, np.pi]  # p,q,r rad/s
    obs[3:6] = [0.0, 0.0, 0.0]  # roll, pitch, yaw rad
    obs[6:9] = [1.0, 0.0, 0.0]  # body lin vel
    obs[9:12] = [0.0, 0.0, 1.0]  # world pos
    st = _obs_to_metric_state(obs)
    np.testing.assert_allclose(st[0:3], [0.0, 0.0, 1.0], atol=1e-5)
    np.testing.assert_allclose(st[3:6], [1.0, 0.0, 0.0], atol=1e-5)  # world=body at level
    np.testing.assert_allclose(st[6:9], [0.0, 0.0, 0.0], atol=1e-5)  # yaw,pitch,roll deg
    np.testing.assert_allclose(st[9:12], [0.0, 0.0, 180.0], atol=1e-3)  # p,q,r deg/s
