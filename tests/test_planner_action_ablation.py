"""Unit tests for predictor action ablation (no PyFlyt)."""

from __future__ import annotations

import torch

from aerojepa.sim.planner import LatentPlanner


class _FakeEnc:
    num_spatial = 4
    num_temporal = 8
    embed_dim = 8

    def __call__(self, clip, ctx):
        n = clip.shape[0]
        return torch.zeros(n, ctx.shape[1], self.embed_dim)


class _FakePred:
    def __call__(self, context_repr, ctx, tgt, acts):
        self.last_acts = acts
        n = context_repr.shape[0]
        return torch.zeros(n, tgt.shape[1], context_repr.shape[-1])


class _FakeModel:
    def __init__(self) -> None:
        self.encoder = _FakeEnc()
        self.predictor = _FakePred()

    def eval(self):
        return self

    def predictor_is_action_conditioned(self) -> bool:
        return True


def test_predictor_ablation_zero_passes_zeros() -> None:
    model = _FakeModel()
    planner = LatentPlanner(
        model,  # type: ignore[arg-type]
        torch.device("cpu"),
        predictor_action_ablation="zero",
    )
    n, t, d = 3, 8, 6
    actions = torch.randn(n, t, d)
    ctx = torch.zeros(n, 16, 8)
    planner._ablation_perm = None
    planner._predict_latents(ctx, context_frames=4, actions_full=actions)
    assert model.predictor.last_acts is not None
    assert torch.equal(model.predictor.last_acts, torch.zeros_like(actions))


def test_predictor_ablation_shuffle_permutes_batch() -> None:
    model = _FakeModel()
    planner = LatentPlanner(
        model,  # type: ignore[arg-type]
        torch.device("cpu"),
        predictor_action_ablation="shuffle",
    )
    n, t, d = 4, 8, 6
    actions = torch.arange(n * t * d, dtype=torch.float32).reshape(n, t, d)
    ctx = torch.zeros(n, 16, 8)
    planner._ablation_perm = None
    planner._predict_latents(ctx, context_frames=4, actions_full=actions)
    got = model.predictor.last_acts
    assert got is not None
    assert got.shape == actions.shape
    # Same multiset of rows (permuted across candidates).
    assert set(float(x) for x in got[:, 0, 0]) == set(float(x) for x in actions[:, 0, 0])
