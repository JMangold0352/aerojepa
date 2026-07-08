"""Train the AeroProber on PyFlyt data with a frozen AeroJEPA checkpoint.

Only the prober (or ablation arm) is trained; the encoder + predictor stay
frozen. Saves prober weights + per-epoch metrics to the output dir.

Usage:
    python research/prober/scripts/train_prober.py \
        --config research/prober/configs/prober_synth.yaml

NOTE: PyFlyt must run OUTSIDE the Cursor sandbox (it segfaults inside). Run this
script with full permissions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

# Make the prober package importable.
_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "research" / "prober" / "src"))
sys.path.insert(0, str(_ROOT / "src"))

from aerojepa_research.prober.data_pyflyt import build_pyflyt_dataloaders
from aerojepa_research.prober.integrator import KinematicIntegrator
from aerojepa_research.prober.loss import (
    NaiveLatentLoss,
    PlainMLPLoss,
    StructuredProberLoss,
    metric_state_mse,
)
from aerojepa_research.prober.prober import PlainMLPHead, Prober
from aerojepa_research.prober.rollout import FrozenRolloutExtractor


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_model_and_loss(cfg: dict, device: torch.device):
    """Build the prober arm + loss for the configured arm type."""
    arm = cfg.get("arm", "structured")  # structured | plain | naive
    integ = KinematicIntegrator(
        dt=cfg["integrator"]["dt"], gravity=cfg["integrator"]["gravity"],
    ).to(device)

    if arm == "structured":
        prober = Prober(
            hidden_dim=cfg["prober"]["hidden_dim"],
            num_layers=cfg["prober"]["num_layers"],
        ).to(device)
        loss_fn = StructuredProberLoss(integ).to(device)
        params = list(prober.parameters())
        return prober, loss_fn, params, arm
    if arm == "plain":
        head = PlainMLPHead(
            hidden_dim=cfg["prober"]["hidden_dim"],
            num_layers=cfg["prober"]["num_layers"],
        ).to(device)
        loss_fn = PlainMLPLoss().to(device)
        params = list(head.parameters())
        return head, loss_fn, params, arm
    if arm == "naive":
        naive = NaiveLatentLoss().to(device)
        params = list(naive.parameters())
        return naive, naive, params, arm
    raise ValueError(f"unknown arm: {arm}")


def compute_loss(arm, model, loss_fn, rollout):
    """Dispatch to the right loss call signature for the arm type."""
    if arm == "structured":
        return loss_fn(model, rollout.latents, rollout.actions, rollout.init_state, rollout.gt_states)
    if arm == "plain":
        return loss_fn(model, rollout.latents, rollout.actions, rollout.init_state, rollout.gt_states)
    if arm == "naive":
        return loss_fn(rollout.latents, rollout.actions, rollout.init_state, rollout.gt_states)
    raise ValueError(arm)


@torch.no_grad()
def evaluate(extractor, model, loss_fn, loader, device, arm, max_loops=None):
    model.eval() if hasattr(model, "eval") else None
    total_loss = 0.0
    n = 0
    for clips, actions, states in loader:
        rollout = extractor.extract(clips, actions, states, max_loops=max_loops)
        loss, _pred = compute_loss(arm, model, loss_fn, rollout)
        total_loss += float(loss.item()) * clips.shape[0]
        n += clips.shape[0]
    return total_loss / max(1, n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", default=None, help="structured | plain | naive (overrides config)")
    parser.add_argument("--seed", type=int, default=None, help="training seed (overrides config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.arm:
        cfg["arm"] = args.arm
    if args.seed is not None:
        cfg["data"]["seed"] = args.seed
    cfg.setdefault("arm", "structured")
    arm = cfg["arm"]

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    torch.manual_seed(cfg["data"]["seed"])
    device = torch.device(cfg.get("device", "mps") if torch.backends.mps.is_available() else "cpu")
    print(f"[prober] device={device} arm={arm} checkpoint={cfg['checkpoint']}")

    # Frozen rollout extractor.
    extractor = FrozenRolloutExtractor(
        cfg["checkpoint"], device=device, context_frames=cfg["train"]["context_frames"],
    )
    max_loops = cfg.get("max_loops") if cfg.get("predictor_mode") == "looped" else None

    # Prober + loss + optimizer.
    model, loss_fn, params, arm = build_model_and_loss(cfg, device)
    n_params = sum(p.numel() for p in params if p.requires_grad)
    print(f"[prober] trainable params: {n_params}")
    opt = torch.optim.AdamW(
        params, lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
    )

    # Data.
    train_loader, val_loader = build_pyflyt_dataloaders(
        batch_size=cfg["data"]["batch_size"],
        num_frames=cfg["data"]["num_frames"],
        img_size=cfg["data"]["img_size"],
        num_train=cfg["data"]["num_train"],
        num_val=cfg["data"]["num_val"],
        num_workers=cfg["data"]["num_workers"],
        seed=cfg["data"]["seed"],
    )

    history = []
    best_val = float("inf")
    for epoch in range(cfg["train"]["epochs"]):
        model.train() if hasattr(model, "train") else None
        epoch_loss = 0.0
        n = 0
        for clips, actions, states in train_loader:
            rollout = extractor.extract(clips, actions, states, max_loops=max_loops)
            loss, _pred = compute_loss(arm, model, loss_fn, rollout)
            opt.zero_grad()
            loss.backward()
            if cfg["train"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(params, cfg["train"]["grad_clip"])
            opt.step()
            epoch_loss += float(loss.item()) * clips.shape[0]
            n += clips.shape[0]

        train_loss = epoch_loss / max(1, n)
        val_loss = evaluate(extractor, model, loss_fn, val_loader, device, arm, max_loops=max_loops)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(f"[prober] epoch {epoch:3d}  train={train_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": (model.state_dict() if hasattr(model, "state_dict") else {}),
                "config": cfg,
                "epoch": epoch,
                "val_loss": val_loss,
            }, out_dir / "best.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[prober] done. best val={best_val:.6f}. saved to {out_dir}")


if __name__ == "__main__":
    main()
