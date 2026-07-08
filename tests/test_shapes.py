from __future__ import annotations

import copy

import torch

from aerojepa.masking import build_mask_collator
from aerojepa.models.jepa import AeroJEPA
from aerojepa.models.looped_predictor import LoopedVideoPredictor
from aerojepa.train import stack_indices
from aerojepa.utils.config import load_config

DEVICE = torch.device("cpu")


def _cfg(**overrides):
    cfg = load_config("configs/smoke_test.yaml")
    for section, values in overrides.items():
        cfg[section] = {**cfg.get(section, {}), **values} if isinstance(values, dict) else values
    return cfg


def _fake_batch(cfg, batch_size=2):
    d = cfg["data"]
    clips = torch.rand(batch_size, d["num_frames"], d.get("in_chans", 3), d["img_size"], d["img_size"])
    return clips


def _run(cfg, actions=None):
    model = AeroJEPA.from_config(cfg).to(DEVICE)
    clips = _fake_batch(cfg).to(DEVICE)
    grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = build_mask_collator(cfg, grid, model.encoder.num_temporal)
    masks = collator(clips.shape[0])
    ctx = stack_indices(masks.context_indices, DEVICE)
    tgt = stack_indices(masks.target_indices, DEVICE)
    out = model(clips, ctx, tgt, actions=actions)
    return model, out, tgt


def test_masked_objective_forward_backward() -> None:
    cfg = _cfg()
    model, out, tgt = _run(cfg)
    assert out["pred_repr"].shape[:2] == tgt.shape
    assert out["pred_repr"].shape[-1] == cfg["encoder"]["embed_dim"]
    out["loss"].backward()  # gradients flow


def test_looped_predictor_returns_exit_probs() -> None:
    cfg = _cfg()  # smoke config has looped + exit gate on
    model, out, _ = _run(cfg)
    assert isinstance(model.predictor, LoopedVideoPredictor)
    assert "exit_probs" in out
    assert out["exit_probs"].shape[1] == cfg["predictor"]["max_loops"]


def test_future_objective() -> None:
    cfg = _cfg()
    cfg["objective"] = "future"
    _, out, tgt = _run(cfg)
    assert out["pred_repr"].shape[:2] == tgt.shape


def test_action_conditioning() -> None:
    cfg = _cfg()
    cfg["predictor"] = {**cfg["predictor"], "action_conditioning": True, "action_dim": 6}
    model = AeroJEPA.from_config(cfg).to(DEVICE)
    assert model.predictor_is_action_conditioned()
    clips = _fake_batch(cfg).to(DEVICE)
    grid = cfg["data"]["img_size"] // cfg["data"]["patch_size"]
    collator = build_mask_collator(cfg, grid, model.encoder.num_temporal)
    masks = collator(clips.shape[0])
    ctx = stack_indices(masks.context_indices, DEVICE)
    tgt = stack_indices(masks.target_indices, DEVICE)
    actions = torch.randn(clips.shape[0], model.encoder.num_temporal, 6)
    out = model(clips, ctx, tgt, actions=actions)
    out["loss"].backward()


def test_tubelet_tokenizer() -> None:
    cfg = _cfg()
    cfg["encoder"] = {**cfg["encoder"], "tokenizer": "tubelet", "tubelet_size": 2}
    model = AeroJEPA.from_config(cfg).to(DEVICE)
    # 4 frames / tubelet 2 -> 2 temporal slots.
    assert model.encoder.num_temporal == 2
    _, out, tgt = _run(cfg)
    assert out["pred_repr"].shape[:2] == tgt.shape


def test_ema_teacher_is_frozen() -> None:
    cfg = _cfg()
    model = AeroJEPA.from_config(cfg)
    assert all(not p.requires_grad for p in model.target_encoder.parameters())
