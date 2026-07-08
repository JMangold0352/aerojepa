from __future__ import annotations

import torch

from aerojepa.data.synthetic import ACTION_DIM, SyntheticDroneClips, render_clip


def test_render_clip_shapes() -> None:
    clip = render_clip(seed=0, num_frames=6, img_size=48, in_chans=3)
    assert clip.frames.shape == (6, 3, 48, 48)
    assert clip.actions.shape == (6, ACTION_DIM)
    assert clip.frames.min() >= 0.0 and clip.frames.max() <= 1.0


def test_render_clip_is_deterministic() -> None:
    a = render_clip(seed=7, num_frames=4, img_size=32)
    b = render_clip(seed=7, num_frames=4, img_size=32)
    assert torch.allclose(a.frames, b.frames)
    assert torch.allclose(a.actions, b.actions)


def test_different_seeds_differ() -> None:
    a = render_clip(seed=1, num_frames=4, img_size=32)
    b = render_clip(seed=2, num_frames=4, img_size=32)
    assert not torch.allclose(a.frames, b.frames)


def test_dataset_indexing() -> None:
    ds = SyntheticDroneClips(num_clips=5, num_frames=4, img_size=32)
    assert len(ds) == 5
    frames, actions = ds[3]
    assert frames.shape == (4, 3, 32, 32)
    assert actions.shape == (4, ACTION_DIM)
