#!/usr/bin/env python
"""SkyJEPA-style compounding + metric RMSE vs horizon.

Arms:
  - latent teacher-forced vs open-loop on uncond / action-cond (v1) JEPA
  - physics-only ControlIntegrator (zero residual) on PyFlyt clips
  - optional structured prober residual if --prober is set

Captions must include horizon, dt, relative-to-t=0. Never compare 0.006 m to
SkyJEPA outdoor 1.43 m.

Example::

    python scripts/eval_compounding.py \\
        --ac-checkpoint checkpoints/action_conditioned_wilds/latest.pt \\
        --uncond-checkpoint checkpoints/real_finetune_fast/latest.pt \\
        --max-batches 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from aerojepa.eval import load_model
from aerojepa.train import _prep_actions, build_dataloaders_from_cfg, build_video_dataloaders
from aerojepa.utils.device import get_device

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "research" / "prober" / "src"))

from aerojepa_research.prober.integrator import ControlIntegrator, MetricState  # noqa: E402
from aerojepa_research.prober.metrics import geodesic_attitude_error_deg  # noqa: E402


@torch.no_grad()
def latent_compounding(
    model,
    loader,
    device: torch.device,
    cfg: dict,
    context_frames: int | None = None,
    max_batches: int = 4,
    use_actions: bool | None = None,
) -> dict:
    """Teacher-forced vs open-loop latent error vs horizon.

    Teacher-forced: each future frame predicted from fixed GT context encoding.
    Open-loop: predicted target tokens are appended to the context representation
    for the next step (autoregressive in latent token space).
    """
    model.eval()
    num_temporal = model.encoder.num_temporal
    num_spatial = model.encoder.num_spatial
    if context_frames is None:
        context_frames = max(1, num_temporal // 2)
    if use_actions is None:
        use_actions = model.predictor_is_action_conditioned()

    horizons = list(range(1, num_temporal - context_frames + 1))
    tf_cos = [0.0] * len(horizons)
    tf_l1 = [0.0] * len(horizons)
    ol_cos = [0.0] * len(horizons)
    ol_l1 = [0.0] * len(horizons)
    n = 0

    frame_slices = [
        torch.arange(f * num_spatial, (f + 1) * num_spatial, device=device)
        for f in range(num_temporal)
    ]

    for bi, (clips, actions) in enumerate(loader):
        if bi >= max_batches:
            break
        clips = clips.to(device)
        B = clips.shape[0]
        acts = _prep_actions(actions, num_temporal, device, cfg) if use_actions else None

        ctx_idx = torch.cat(frame_slices[:context_frames]).unsqueeze(0).expand(B, -1)
        context_repr = model.encoder(clips, ctx_idx)

        # Teacher targets from EMA for all future frames.
        with torch.no_grad():
            all_tokens = model.target_encoder.forward_all_patches(clips)

        # --- Teacher-forced (independent horizons from fixed context) ---
        for hi, h in enumerate(horizons):
            f = context_frames + h - 1
            tgt = frame_slices[f].unsqueeze(0).expand(B, -1)
            pred = model.predictor(context_repr, ctx_idx, tgt, acts)
            if isinstance(pred, tuple):
                pred = pred[0]
            target = torch.gather(
                all_tokens, 1, tgt.unsqueeze(-1).expand(-1, -1, all_tokens.size(-1))
            )
            tf_cos[hi] += float(F.cosine_similarity(pred, target, dim=-1).mean())
            tf_l1[hi] += float(F.smooth_l1_loss(pred, target))

        # --- Open-loop (grow context with predictions) ---
        grow_repr = context_repr
        grow_idx = ctx_idx
        for hi, h in enumerate(horizons):
            f = context_frames + h - 1
            tgt = frame_slices[f].unsqueeze(0).expand(B, -1)
            pred = model.predictor(grow_repr, grow_idx, tgt, acts)
            if isinstance(pred, tuple):
                pred = pred[0]
            target = torch.gather(
                all_tokens, 1, tgt.unsqueeze(-1).expand(-1, -1, all_tokens.size(-1))
            )
            ol_cos[hi] += float(F.cosine_similarity(pred, target, dim=-1).mean())
            ol_l1[hi] += float(F.smooth_l1_loss(pred, target))
            # Append predicted tokens as next context.
            grow_repr = torch.cat([grow_repr, pred], dim=1)
            grow_idx = torch.cat([grow_idx, tgt], dim=1)

        n += 1

    n = max(1, n)
    tf_l1_n = [x / n for x in tf_l1]
    ol_l1_n = [x / n for x in ol_l1]
    # Compounding ratio on L1 (SkyJEPA-style: open / teacher-forced); clamp den.
    cr = [o / max(t, 1e-8) for o, t in zip(ol_l1_n, tf_l1_n)]
    er = [ol_l1_n[0]] + [ol_l1_n[i] - ol_l1_n[i - 1] for i in range(1, len(ol_l1_n))]
    return {
        "horizon": horizons,
        "teacher_forced": {"cosine": [c / n for c in tf_cos], "smooth_l1": tf_l1_n},
        "open_loop": {"cosine": [c / n for c in ol_cos], "smooth_l1": ol_l1_n},
        "compounding_ratio_l1": cr,
        "error_rate_l1": er,
        "context_frames": context_frames,
        "n_batches": n,
    }


@torch.no_grad()
def physics_only_metric_vs_horizon(
    clips_controls_states: list[tuple[torch.Tensor, torch.Tensor]],
    dt: float = 0.025,
    context_frames: int = 4,
) -> dict:
    """Zero-residual ControlIntegrator rollout; pos/geodesic-att vs horizon.

    Position errors are relative to t=0 of the *predict* window (init = last
    context state). Horizon h is h * dt seconds.
    """
    integ = ControlIntegrator(dt=dt)
    pos_rmse: dict[int, list[float]] = {}
    att_geo: dict[int, list[float]] = {}

    for controls, metric_states in clips_controls_states:
        # controls, metric_states: (B, T, *)
        B, T, _ = metric_states.shape
        init = MetricState(
            pos=metric_states[:, context_frames - 1, 0:3],
            vel=metric_states[:, context_frames - 1, 3:6],
            euler_att=metric_states[:, context_frames - 1, 6:9],
            ang_vel=metric_states[:, context_frames - 1, 9:12],
        )
        fut_ctrl = controls[:, context_frames:T]
        H = fut_ctrl.shape[1]
        z_lin = torch.zeros(B, H, 3, device=controls.device)
        z_ang = torch.zeros(B, H, 3, device=controls.device)
        pred = integ.rollout(init, fut_ctrl, z_lin, z_ang)
        gt = metric_states[:, context_frames:T]
        # Relative position: subtract init pos from both.
        pred_pos_rel = pred.pos - init.pos.unsqueeze(1)
        gt_pos_rel = gt[..., 0:3] - init.pos.unsqueeze(1)
        for h in range(H):
            pe = torch.sqrt(((pred_pos_rel[:, h] - gt_pos_rel[:, h]) ** 2).sum(-1))
            ae = geodesic_attitude_error_deg(pred.euler_att[:, h], gt[:, h, 6:9])
            pos_rmse.setdefault(h + 1, []).extend(pe.cpu().tolist())
            att_geo.setdefault(h + 1, []).extend(ae.cpu().tolist())

    horizons = sorted(pos_rmse)
    return {
        "dt": dt,
        "horizon_frames": horizons,
        "horizon_seconds": [h * dt for h in horizons],
        "position_relative_to": "predict_window_t0 (last context state)",
        "pos_rmse_m": [float(np.sqrt(np.mean(np.square(pos_rmse[h])))) for h in horizons],
        "att_geodesic_rmse_deg": [
            float(np.sqrt(np.mean(np.square(att_geo[h])))) for h in horizons
        ],
        "arm": "physics_only_zero_residual",
    }


def _plot(report: dict, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    arms = report.get("latent_arms", {})
    for name, arm in arms.items():
        h = arm["horizon"]
        axes[0].plot(h, arm["compounding_ratio_l1"], marker="o", label=name)
        axes[1].plot(h, arm["open_loop"]["smooth_l1"], marker="o", label=f"{name} OL")
        axes[1].plot(
            h, arm["teacher_forced"]["smooth_l1"], marker="x", linestyle="--", label=f"{name} TF"
        )
    axes[0].axhline(1.0, color="gray", lw=0.8)
    axes[0].set_xlabel("horizon (frames)")
    axes[0].set_ylabel("CR = L1_OL / L1_TF")
    axes[0].set_title("Compounding ratio")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("horizon (frames)")
    axes[1].set_ylabel("smooth-L1")
    axes[1].set_title("Latent error vs horizon")
    axes[1].legend(fontsize=7)

    phys = report.get("physics_only")
    if phys:
        hs = phys["horizon_seconds"]
        axes[2].plot(hs, phys["pos_rmse_m"], marker="o", label="pos RMSE (m)")
        ax2b = axes[2].twinx()
        ax2b.plot(hs, phys["att_geodesic_rmse_deg"], marker="s", color="C1", label="att geo (°)")
        axes[2].set_xlabel("horizon (s)")
        axes[2].set_ylabel("pos RMSE (m), relative to t=0")
        ax2b.set_ylabel("geodesic att RMSE (°)")
        axes[2].set_title(f"Physics-only (dt={phys['dt']} s)")
        axes[2].legend(loc="upper left", fontsize=7)
        ax2b.legend(loc="upper right", fontsize=7)
    else:
        axes[2].text(0.5, 0.5, "physics-only\nnot run", ha="center", va="center")
        axes[2].axis("off")

    fig.suptitle(
        "Compounding / metric vs horizon - relative pos; do not compare to SkyJEPA outdoor RMSE",
        fontsize=9,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ac-checkpoint",
        default="checkpoints/action_conditioned_wilds/latest.pt",
    )
    parser.add_argument(
        "--uncond-checkpoint",
        default="checkpoints/real_finetune_fast/latest.pt",
    )
    parser.add_argument("--data-dir", default="data/flights_128")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument(
        "--pyflyt-clips",
        type=int,
        default=8,
        help="Number of on-the-fly PyFlyt clips for physics-only overlay (0=skip).",
    )
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--out", default="results/compounding.json")
    parser.add_argument(
        "--figure",
        default="visualizations/figures/compounding_vs_horizon.png",
    )
    args = parser.parse_args()

    device = get_device(args.device)
    report: dict = {
        "protocol": "compounding_v1",
        "caption_notes": {
            "position": "relative to predict-window t=0 (last context state)",
            "dt": args.dt,
            "do_not_compare": "AeroProber ~0.006 m vs SkyJEPA outdoor 1.43 m",
        },
        "latent_arms": {},
    }

    for label, ckpt in (
        ("action_cond_v1", args.ac_checkpoint),
        ("uncond", args.uncond_checkpoint),
    ):
        path = Path(ckpt)
        if not path.is_file():
            print(f"skip {label}: missing {ckpt}")
            continue
        model, cfg = load_model(ckpt, device)
        data_cfg = dict(cfg["data"])
        if Path(args.data_dir).is_dir():
            data_cfg["source"] = "video"
            data_cfg["data_dir"] = args.data_dir
            data_cfg["window_mode"] = "uniform"
            data_cfg["batch_size"] = min(int(data_cfg.get("batch_size", 4)), 4)
            data_cfg["num_workers"] = 0
            _, loader = build_video_dataloaders(data_cfg)
        else:
            _, loader = build_dataloaders_from_cfg(cfg)
        arm = latent_compounding(
            model, loader, device, cfg, max_batches=args.max_batches
        )
        arm["checkpoint"] = ckpt
        report["latent_arms"][label] = arm
        print(
            f"{label}: CR@H={[f'{c:.2f}' for c in arm['compounding_ratio_l1']]} "
            f"OL_L1={[f'{x:.4f}' for x in arm['open_loop']['smooth_l1']]}"
        )

    if args.pyflyt_clips > 0:
        try:
            from aerojepa_research.prober.data_pyflyt import generate_clip

            print(f"Generating {args.pyflyt_clips} PyFlyt clips for physics-only…")
            pairs = []
            for i in range(args.pyflyt_clips):
                c = generate_clip(
                    seed=1000 + i,
                    num_frames=8,
                    img_size=64,
                )
                ctrl = c.control_actions.unsqueeze(0).float()
                st = c.metric_state.unsqueeze(0).float()
                pairs.append((ctrl, st))
            report["physics_only"] = physics_only_metric_vs_horizon(
                pairs, dt=args.dt, context_frames=4
            )
            po = report["physics_only"]
            print(
                "physics-only pos RMSE (m):",
                [f"{x:.4f}" for x in po["pos_rmse_m"]],
                "horizons (s):",
                po["horizon_seconds"],
            )
        except Exception as exc:  # noqa: BLE001 - report and continue
            report["physics_only_error"] = str(exc)
            print(f"physics-only skipped: {exc}")

    fig_path = Path(args.figure)
    _plot(report, fig_path)
    report["figure"] = str(fig_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_path}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
