from __future__ import annotations

import torch

from aerojepa.eval import load_model
from aerojepa.eval.latent_pred import latent_prediction_metrics
from aerojepa.eval.rollout import rollout_metrics
from aerojepa.masking import build_mask_collator
from aerojepa.models.jepa import AeroJEPA
from aerojepa.train import build_dataloaders_from_cfg, load_pretrained_weights, train
from aerojepa.utils.config import load_config


def test_train_smoke_end_to_end(tmp_path) -> None:
    cfg = load_config("configs/smoke_test.yaml")
    cfg["train"]["run_dir"] = str(tmp_path / "runs")
    cfg["train"]["checkpoint_dir"] = str(tmp_path / "ckpt")

    ckpt = train(cfg, torch.device("cpu"))
    assert ckpt.exists()

    # Reload and evaluate the freshly trained model.
    model, loaded_cfg = load_model(str(ckpt), torch.device("cpu"))
    _, val_loader = build_dataloaders_from_cfg(loaded_cfg)
    grid = loaded_cfg["data"]["img_size"] // loaded_cfg["data"]["patch_size"]
    collator = build_mask_collator(loaded_cfg, grid, model.encoder.num_temporal)

    latent = latent_prediction_metrics(model, val_loader, collator, torch.device("cpu"), loaded_cfg, max_batches=1)
    assert -1.0 <= latent["cosine"] <= 1.0

    roll = rollout_metrics(model, val_loader, torch.device("cpu"), loaded_cfg, max_batches=1)
    assert len(roll["cosine"]) == len(roll["horizon"]) > 0


def test_init_checkpoint_warm_start(tmp_path) -> None:
    """--init-checkpoint should copy trained weights into a fresh model."""
    cfg = load_config("configs/smoke_test.yaml")
    cfg["train"]["run_dir"] = str(tmp_path / "runs")
    cfg["train"]["checkpoint_dir"] = str(tmp_path / "ckpt")
    ckpt = train(cfg, torch.device("cpu"))

    fresh = AeroJEPA.from_config(cfg)
    trained, _ = load_model(str(ckpt), torch.device("cpu"))
    differs_before = any(
        not torch.allclose(fresh.state_dict()[name], tensor)
        for name, tensor in trained.state_dict().items()
    )
    assert differs_before, "fresh and trained models should start out different"

    load_pretrained_weights(fresh, str(ckpt), torch.device("cpu"))
    for name, tensor in trained.state_dict().items():
        assert torch.allclose(fresh.state_dict()[name], tensor)


def test_resume_checkpoint_continues_training(tmp_path) -> None:
    """--resume should restore optimizer state and skip completed epochs."""
    cfg = load_config("configs/smoke_test.yaml")
    cfg["train"]["run_dir"] = str(tmp_path / "runs")
    cfg["train"]["checkpoint_dir"] = str(tmp_path / "ckpt")
    cfg["train"]["epochs"] = 1

    first = train(cfg, torch.device("cpu"))
    assert first.exists()

    cfg["train"]["epochs"] = 2
    resumed = train(cfg, torch.device("cpu"), resume_checkpoint=str(first))
    assert resumed.exists()

    ckpt = torch.load(resumed, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 2
    assert ckpt["step"] > 0
