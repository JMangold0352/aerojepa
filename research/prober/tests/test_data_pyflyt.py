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


def test_generate_clip_reproducible():
    from aerojepa_research.prober.data_pyflyt import generate_clip

    c1 = generate_clip(seed=42, num_frames=8, img_size=64)
    c2 = generate_clip(seed=42, num_frames=8, img_size=64)
    assert c1.frames.equal(c2.frames)
    assert c1.actions.equal(c2.actions)
    assert c1.metric_state.equal(c2.metric_state)


def test_states_to_actions_matches_telemetry_convention():
    """Linear channels must be the per-frame velocity (not a delta), matching
    ``telemetry.derive_actions_from_raw`` (which copies vgx/vgy/vgz for every
    row). Angular channels are wrapped deltas with the first row zero.
    """
    from aerojepa_research.prober.data_pyflyt import states_to_actions
    import numpy as np

    # 5 frames, distinct velocities so we can distinguish velocity from delta.
    states = np.zeros((5, 12), dtype=np.float32)
    states[:, 3:6] = np.arange(5).reshape(-1, 1) * 0.5  # vel = 0, 0.5, 1.0, 1.5, 2.0
    # Vary attitude so angular deltas are non-trivial.
    states[:, 6] = [10.0, 25.0, 45.0, 80.0, 120.0]  # yaw (deg)

    actions = states_to_actions(states)
    assert actions.shape == (5, 6)
    # Linear channels = velocity itself, for EVERY row (including row 0).
    np.testing.assert_allclose(actions[:, 0:3], states[:, 3:6], atol=1e-6)
    # Angular channel row 0 is zero (no previous frame).
    assert np.allclose(actions[0, 3:6], 0.0)
    # Angular channels are wrapped deltas for rows 1+.
    assert abs(actions[1, 3] - 15.0) < 1e-5  # 25 - 10
    assert abs(actions[2, 3] - 20.0) < 1e-5  # 45 - 25
