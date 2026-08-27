#!/usr/bin/env python
"""Train a tiny ActionResidualHead on top of a frozen AeroJEPA planner.

Supervised objective: match PyFlyt ground-truth controls given
(heuristic map of AeroJEPA actions + imagined/context latents), while keeping
the residual small.

    û = clip( heuristic(a_6) + residual(z, heuristic(a_6), a_6) )
    L = ||û - u_gt||² + λ ||residual||²

The world model stays frozen; only the residual MLP (~1-3k params) is trained.

Example::

    python scripts/train_action_residual.py \\
        --checkpoint checkpoints/action_conditioned/latest.pt \\
        --epochs 5 --num-train 128 --output-dir checkpoints/action_residual

Requires PyFlyt (run outside the Cursor sandbox).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "research" / "prober" / "src"))

from aerojepa.eval import load_model
from aerojepa.sim.action_residual import (
    ActionResidualHead,
    residual_loss,
    save_residual_checkpoint,
)
from aerojepa.utils.device import get_device


def _pool_spatial(latents: torch.Tensor, num_spatial: int) -> torch.Tensor:
    """(B, T*S, D) or (B, T, S, D) → (B, T, D)."""
    if latents.dim() == 4:
        return latents.mean(dim=2)
    b, n, d = latents.shape
    if n % num_spatial != 0:
        raise ValueError(f"cannot pool n={n} with num_spatial={num_spatial}")
    t = n // num_spatial
    return latents.view(b, t, num_spatial, d).mean(dim=2)


@torch.no_grad()
def _extract_future_latents(
    model,
    clips: torch.Tensor,
    actions: torch.Tensor,
    context_frames: int,
) -> torch.Tensor:
    """Predict future-frame latents with the frozen world model.

    ``clips``: (B, T, C, H, W), ``actions``: (B, T, 6)
    Returns pooled latents (B, T_pred, D) for frames ``context_frames:``.
    """
    device = next(model.parameters()).device
    clips = clips.to(device)
    actions = actions.to(device)
    b, t, _, _, _ = clips.shape
    num_spatial = model.encoder.num_spatial
    num_temporal = model.encoder.num_temporal
    if t != num_temporal:
        # Pad / crop to model temporal length.
        if t < num_temporal:
            pad = clips[:, -1:].expand(b, num_temporal - t, -1, -1, -1)
            clips = torch.cat([clips, pad], dim=1)
            apad = torch.zeros(b, num_temporal - t, actions.shape[-1], device=device)
            actions = torch.cat([actions, apad], dim=1)
        else:
            clips = clips[:, :num_temporal]
            actions = actions[:, :num_temporal]

    ctx = torch.arange(0, context_frames * num_spatial, device=device).unsqueeze(0).expand(b, -1)
    tgt = torch.arange(
        context_frames * num_spatial, num_temporal * num_spatial, device=device
    ).unsqueeze(0).expand(b, -1)
    acts = actions if model.predictor_is_action_conditioned() else None
    out = model(clips, ctx, tgt, actions=acts)
    pred = out["pred_repr"]  # (B, T_pred * S, D)
    return _pool_spatial(pred, num_spatial)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/action_conditioned/latest.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-train", type=int, default=128)
    parser.add_argument("--num-val", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--residual-l2", type=float, default=0.1)
    parser.add_argument("--hover-thrust", type=float, default=0.39)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="checkpoints/action_residual")
    parser.add_argument(
        "--wind-mps",
        type=float,
        default=2.0,
        help="Minimum wind speed (m/s) for the wind-augmented clip fraction.",
    )
    parser.add_argument(
        "--wind-mps-max",
        type=float,
        default=4.0,
        help="Maximum wind speed (m/s) sampled for wind clips.",
    )
    parser.add_argument(
        "--wind-fraction",
        type=float,
        default=0.5,
        help="Fraction of clips generated under wind with a hover/counter policy.",
    )
    parser.add_argument(
        "--kick-fraction",
        type=float,
        default=0.0,
        help="Fraction of clips with a lateral kick then brake/home (recover).",
    )
    parser.add_argument(
        "--turn-fraction",
        type=float,
        default=0.0,
        help="Fraction of clips that seek an L-path (corner supervision).",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    torch.manual_seed(args.seed)

    model, cfg = load_model(args.checkpoint, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    latent_dim = int(cfg["encoder"]["embed_dim"])
    print(
        f"[residual] frozen={args.checkpoint} latent_dim={latent_dim} "
        f"action_cond={model.predictor_is_action_conditioned()}"
    )
    print(
        f"[residual] mix wind={args.wind_fraction} kick={args.kick_fraction} "
        f"turn={args.turn_fraction}  wind=[{args.wind_mps}, {args.wind_mps_max}] m/s"
    )

    from aerojepa_research.prober.data_pyflyt import build_pyflyt_dataloaders

    train_loader, val_loader = build_pyflyt_dataloaders(
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        img_size=args.img_size,
        num_train=args.num_train,
        num_val=args.num_val,
        num_workers=0,
        seed=args.seed,
        wind_mps=args.wind_mps,
        wind_mps_max=args.wind_mps_max,
        wind_fraction=args.wind_fraction,
        kick_fraction=args.kick_fraction,
        turn_fraction=args.turn_fraction,
        flight_dome_size=8.0,
    )

    head = ActionResidualHead(
        latent_dim=latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    print(f"[residual] trainable params: {head.num_params()}")
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_val = float("inf")

    def run_epoch(loader, train: bool) -> dict[str, float]:
        head.train(train)
        totals = {"loss": 0.0, "mse": 0.0, "mse_gt": 0.0, "heur_mse": 0.0, "delta_l2": 0.0}
        n = 0
        for clips, actions, _states, controls in loader:
            # clips (B,T,C,H,W), actions (B,T,6), controls (B,T,4)
            with torch.no_grad():
                latents = _extract_future_latents(
                    model, clips, actions, args.context_frames
                )
            aero = actions[:, args.context_frames :].to(device)
            gt = controls[:, args.context_frames :].to(device)
            # Align lengths if predictor horizon differs.
            t = min(latents.shape[1], aero.shape[1], gt.shape[1])
            latents, aero, gt = latents[:, :t], aero[:, :t], gt[:, :t]

            if train:
                loss, stats = residual_loss(
                    head,
                    latents,
                    aero,
                    gt,
                    hover_thrust=args.hover_thrust,
                    residual_l2=args.residual_l2,
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                opt.step()
            else:
                with torch.no_grad():
                    _, stats = residual_loss(
                        head,
                        latents,
                        aero,
                        gt,
                        hover_thrust=args.hover_thrust,
                        residual_l2=args.residual_l2,
                    )
            bs = clips.shape[0]
            for k in totals:
                totals[k] += stats[k] * bs
            n += bs
        return {k: v / max(n, 1) for k, v in totals.items()}

    for epoch in range(args.epochs):
        tr = run_epoch(train_loader, train=True)
        va = run_epoch(val_loader, train=False)
        row = {"epoch": epoch, "train": tr, "val": va}
        history.append(row)
        print(
            f"[residual] epoch {epoch:3d}  "
            f"train_mse={tr['mse']:.5f} gt={tr['mse_gt']:.5f} (heur={tr['heur_mse']:.5f})  "
            f"val_mse={va['mse']:.5f} gt={va['mse_gt']:.5f} (heur={va['heur_mse']:.5f})"
        )
        if va["mse"] < best_val:
            best_val = va["mse"]
            save_residual_checkpoint(
                out_dir / "best.pt",
                head,
                config={
                    "latent_dim": latent_dim,
                    "hidden_dim": args.hidden_dim,
                    "hover_thrust": args.hover_thrust,
                    "residual_l2": args.residual_l2,
                    "world_checkpoint": args.checkpoint,
                    "wind_mps": args.wind_mps,
                    "wind_mps_max": args.wind_mps_max,
                    "wind_fraction": args.wind_fraction,
                    "kick_fraction": args.kick_fraction,
                    "turn_fraction": args.turn_fraction,
                },
                epoch=epoch,
                metrics=va,
            )

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"[residual] done. best val_mse={best_val:.5f} → {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
