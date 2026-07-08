"""Integration test: frozen-model rollout extraction on a real checkpoint.

Verifies that:
- A checkpoint loads and the model freezes (no trainable params).
- The extractor produces correctly-shaped per-frame latent rollouts.
- max_loops=1 on a looped checkpoint behaves as single-pass.
- The whole pipeline is differentiable through the prober to the loss.

Slow + requires PyFlyt + a checkpoint. Run OUTSIDE the sandbox:
    pytest research/prober/tests/test_rollout.py -v -s
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.integration]


@pytest.fixture(scope="module")
def device():
    import torch
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def test_load_and_freeze(device):
    from aerojepa_research.prober.rollout import FrozenRolloutExtractor

    ext = FrozenRolloutExtractor(
        "checkpoints/baseline/latest.pt", device=device, context_frames=4,
    )
    trainable = sum(p.numel() for p in ext.model.parameters() if p.requires_grad)
    assert trainable == 0, "frozen model should have zero trainable params"
    assert ext.num_temporal == 8
    assert ext.num_pred_frames == 4
    assert ext.encoder_dim == 192


def test_extract_shapes(device):
    from aerojepa_research.prober.rollout import FrozenRolloutExtractor
    from aerojepa_research.prober.data_pyflyt import generate_clip

    ext = FrozenRolloutExtractor(
        "checkpoints/baseline/latest.pt", device=device, context_frames=4,
    )
    clip = generate_clip(seed=0, num_frames=8, img_size=64)
    clips = clip.frames.unsqueeze(0).to(device)
    actions = clip.actions.unsqueeze(0).to(device)
    states = clip.metric_state.unsqueeze(0).to(device)

    rollout = ext.extract(clips, actions, states)
    assert rollout.latents.shape == (1, 4, 192)
    assert rollout.actions.shape == (1, 4, 6)
    assert rollout.gt_states.shape == (1, 4, 12)
    assert rollout.init_state.pos.shape == (1, 3)
    # Latents should be finite (not NaN).
    assert rollout.latents.isfinite().all()


def test_looped_checkpoint_max_loops_1(device):
    """A looped checkpoint with max_loops=1 should produce valid latents too."""
    from aerojepa_research.prober.rollout import FrozenRolloutExtractor
    from aerojepa_research.prober.data_pyflyt import generate_clip

    ext = FrozenRolloutExtractor(
        "checkpoints/world_model/latest.pt", device=device, context_frames=4,
    )
    assert ext.is_looped()
    clip = generate_clip(seed=1, num_frames=8, img_size=64)
    clips = clip.frames.unsqueeze(0).to(device)
    actions = clip.actions.unsqueeze(0).to(device)
    states = clip.metric_state.unsqueeze(0).to(device)

    rollout_regular = ext.extract(clips, actions, states, max_loops=1)
    rollout_looped = ext.extract(clips, actions, states, max_loops=3)
    assert rollout_regular.latents.isfinite().all()
    assert rollout_looped.latents.isfinite().all()
    # max_loops=1 vs max_loops=3 should generally differ (more refinement).
    assert not torch.allclose(rollout_regular.latents, rollout_looped.latents)


def test_full_pipeline_differentiable(device):
    """End-to-end: PyFlyt -> frozen encoder -> frozen predictor -> prober -> integrator -> loss.

    Gradients must flow ONLY to the prober (the frozen model has no grad).
    """
    from aerojepa_research.prober.rollout import FrozenRolloutExtractor
    from aerojepa_research.prober.data_pyflyt import generate_clip
    from aerojepa_research.prober import Prober, KinematicIntegrator

    ext = FrozenRolloutExtractor(
        "checkpoints/baseline/latest.pt", device=device, context_frames=4,
    )
    prober = Prober().to(device)
    integ = KinematicIntegrator(dt=1.0 / 15.0).to(device)

    clip = generate_clip(seed=2, num_frames=8, img_size=64)
    clips = clip.frames.unsqueeze(0).to(device)
    actions = clip.actions.unsqueeze(0).to(device)
    states = clip.metric_state.unsqueeze(0).to(device)

    rollout = ext.extract(clips, actions, states)
    res_lin, res_ang = prober(rollout.latents, rollout.actions)
    pred_traj = integ.rollout(rollout.init_state, rollout.actions, res_lin, res_ang)
    pred_stack = pred_traj.stack()  # (1, T_pred, 12)
    loss = ((pred_stack - rollout.gt_states) ** 2).mean()
    loss.backward()

    # Prober must have gradients.
    for p in prober.parameters():
        assert p.grad is not None
        assert torch.any(p.grad != 0)
    # Frozen model must NOT have gradients.
    for p in ext.model.parameters():
        assert p.grad is None or torch.all(p.grad == 0)
