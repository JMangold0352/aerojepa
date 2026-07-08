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


def test_states_to_actions_first_row_zero():
    from aerojepa_research.prober.data_pyflyt import states_to_actions
    import numpy as np

    states = np.random.randn(5, 12).astype(np.float32)
    actions = states_to_actions(states)
    assert np.allclose(actions[0], 0.0)
    assert actions.shape == (5, 6)
