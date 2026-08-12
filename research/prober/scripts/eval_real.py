"""Real-data evaluation on Parrot Wilds (partial-quantitative).

Loads a trained structured prober + a frozen AeroJEPA checkpoint, runs the
prober on real Wilds clips, and reports quantitative metrics where ground truth
exists: velocity RMSE, attitude RMSE, altitude RMSE. Position x/y is
dead-reckoned (no GPS) so it is reported qualitatively only.

Leak-free control input: Parrot logs lack motor commands. Default is **zero
controls**; optional ``--controls hover`` uses a constant exogenous thrust
prior (not GT velocity / pose-delta actions).

Gap-closure eval protocol (v4):
  * Contiguous frame windows (not uniform spanning of the whole clip).
  * Integrator ``dt = 1/fps`` matching the video timeline.
  * GT body-frame velocity rotated to world before RMSE (matches integrator).

Usage:
    python research/prober/scripts/eval_real.py \\
        --prober research/prober/results/prober_real_finetune/best.pt \\
        --checkpoint checkpoints/real_finetune_fast/latest.pt \\
        --data-dir data/flights_with_state \\
        --num-clips 16

NOTE: runs OUTSIDE the Cursor sandbox (needs video decoding).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "research" / "prober" / "src"))
sys.path.insert(0, str(_ROOT / "src"))

from aerojepa_research.prober.integrator import ControlIntegrator
from aerojepa_research.prober.metrics import compute_metrics, metrics_to_dict
from aerojepa_research.prober.prober import Prober
from aerojepa_research.prober.pseudo_controls import (
    make_eval_controls,
    metric_stack_body_vel_to_world,
)
from aerojepa_research.prober.rollout import FrozenRolloutExtractor


def load_clip_with_state(
    video_path: Path,
    state_csv: Path,
    num_frames: int,
    img_size: int,
    device,
    *,
    controls_mode: str = "zeros",
    hover_thrust: float = 0.39,
    seed: int = 0,
    body_to_world: bool = True,
):
    """Load one contiguous clip window + metric state CSV.

    Returns clips, controls, metric state, and the per-frame ``dt`` (seconds).
    """
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 1e-3:
        fps = 15.0
    if total < num_frames:
        cap.release()
        return None

    # Contiguous window: keeps integrator dt honest (vs spanning the whole clip).
    rng = np.random.default_rng(seed)
    max_start = total - num_frames
    start = int(rng.integers(0, max_start + 1)) if max_start > 0 else 0
    idxs = np.arange(start, start + num_frames, dtype=int)

    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame[:, :, :3])
    cap.release()
    clips = torch.from_numpy(np.stack(frames)).float() / 255.0
    clips = clips.permute(0, 3, 1, 2)

    state_all = np.loadtxt(state_csv, delimiter=",", skiprows=1, ndmin=2).astype(np.float32)
    state_rows = np.zeros((num_frames, 12), dtype=np.float32)
    for slot, fi in enumerate(idxs):
        state_rows[slot] = state_all[min(int(fi), len(state_all) - 1)]
    metric_states = torch.from_numpy(state_rows)
    if body_to_world:
        # Parrot velocities are body-frame; ControlIntegrator / PyFlyt GT are world-frame.
        metric_states = metric_stack_body_vel_to_world(metric_states)

    controls = make_eval_controls(
        num_frames,
        mode=controls_mode,
        hover_thrust=hover_thrust,
        device="cpu",
    )
    dt = 1.0 / fps
    return clips.to(device), controls.to(device), metric_states.to(device), dt, fps, start


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prober", required=True, help="path to trained prober best.pt")
    parser.add_argument("--checkpoint", default="checkpoints/real_finetune_fast/latest.pt")
    parser.add_argument("--data-dir", default="data/flights_with_state")
    parser.add_argument("--num-clips", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument(
        "--integrator-dt",
        type=float,
        default=None,
        help="Override dt. Default: 1/fps per clip (contiguous protocol).",
    )
    parser.add_argument(
        "--controls",
        default="zeros",
        choices=["zeros", "hover"],
        help="Eval control prior (leak-free). Default zeros.",
    )
    parser.add_argument("--hover-thrust", type=float, default=0.39)
    parser.add_argument(
        "--gravity",
        type=float,
        default=None,
        help="Override integrator gravity. Default: -9.81, or 0.0 when --controls zeros "
        "(weaker nominal for unknown motors; see note.md §8).",
    )
    parser.add_argument(
        "--no-body-to-world",
        action="store_true",
        help="Disable body→world rotation of GT velocity (legacy apples-to-oranges).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="research/prober/results/real_data_v4")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[real] device={device} checkpoint={args.checkpoint}")

    extractor = FrozenRolloutExtractor(
        args.checkpoint, device=device, context_frames=args.context_frames,
    )
    img_size = extractor.cfg["data"]["img_size"]
    print(f"[real] img_size={img_size} looped={extractor.is_looped()}")

    prober = Prober().to(device)
    ckpt = torch.load(args.prober, map_location=device, weights_only=False)
    # Older checkpoints lack ang_residual_scale buffer — load with defaults.
    missing, unexpected = prober.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        print(f"[real] state_dict notes missing={missing} unexpected={unexpected}")
    # Prefer scale stored in training config when present.
    cfg = ckpt.get("config") or {}
    scale = float((cfg.get("prober") or {}).get("ang_residual_scale", 0.25))
    with torch.no_grad():
        prober.ang_residual_scale.fill_(scale)
    prober.eval()

    data_dir = Path(args.data_dir)
    clip_paths = sorted(data_dir.glob("wilds_*.mp4"))[: args.num_clips]
    print(
        f"[real] found {len(clip_paths)} clips | controls={args.controls} | "
        f"body→world={not args.no_body_to_world}"
    )

    per_clip = []
    all_pred_vel, all_gt_vel = [], []
    all_pred_att, all_gt_att = [], []
    for ci, vp in enumerate(clip_paths):
        state_csv = data_dir / (vp.stem + "_state.csv")
        if not state_csv.exists():
            print(f"[real] clip {vp.stem}: no state csv, skipping")
            continue
        batch = load_clip_with_state(
            vp,
            state_csv,
            args.num_frames,
            img_size,
            device,
            controls_mode=args.controls,
            hover_thrust=args.hover_thrust,
            seed=args.seed + ci,
            body_to_world=not args.no_body_to_world,
        )
        if batch is None:
            continue
        clips_t, controls_t, states_t, dt_clip, fps, start = batch
        dt = float(args.integrator_dt) if args.integrator_dt is not None else float(dt_clip)
        if args.gravity is not None:
            gravity = float(args.gravity)
        elif args.controls == "zeros":
            # Weaker nominal when motors are unknown (note.md §8): do not force
            # free-fall under T=0 — residual + init state carry dynamics.
            gravity = 0.0
        else:
            gravity = -9.81
        integ = ControlIntegrator(
            dt=dt,
            gravity=gravity,
            mass=1.0,
            hover_thrust=args.hover_thrust,
        ).to(device)

        clips_b = clips_t.unsqueeze(0)
        controls_b = controls_t.unsqueeze(0)
        states_b = states_t.unsqueeze(0)

        with torch.no_grad():
            rollout = extractor.extract(clips_b, controls_b, states_b, max_loops=1)
            res_lin, res_ang = prober(rollout.latents, rollout.controls)
            pred_traj = integ.rollout(rollout.init_state, rollout.controls, res_lin, res_ang)
            pred_stack = pred_traj.stack()
            gt_stack = rollout.gt_states

        m = compute_metrics(pred_stack, gt_stack)
        alt_rmse = float(
            np.sqrt(
                (
                    (pred_stack[0, :, 2].cpu().numpy() - rollout.gt_states[0, :, 2].cpu().numpy())
                    ** 2
                ).mean()
            )
        )
        per_clip.append(
            {
                "clip": vp.stem,
                "altitude_rmse": alt_rmse,
                "dt": dt,
                "fps": fps,
                "window_start": int(start),
                **metrics_to_dict(m),
            }
        )
        all_pred_vel.append(pred_stack[0, :, 3:6].cpu().numpy())
        all_gt_vel.append(gt_stack[0, :, 3:6].cpu().numpy())
        all_pred_att.append(pred_stack[0, :, 6:9].cpu().numpy())
        all_gt_att.append(gt_stack[0, :, 6:9].cpu().numpy())
        print(
            f"[real] clip {vp.stem}: vel_rmse={m.velocity_rmse:.4f} "
            f"att_rmse={m.attitude_rmse_deg:.4f} alt_rmse={alt_rmse:.4f} "
            f"dt={dt:.4f} start={start}"
        )

    if per_clip:
        agg = {
            "n_clips": len(per_clip),
            "velocity_rmse_mean": float(np.mean([r["velocity_rmse"] for r in per_clip])),
            "velocity_rmse_std": float(np.std([r["velocity_rmse"] for r in per_clip])),
            "attitude_rmse_deg_mean": float(np.mean([r["attitude_rmse_deg"] for r in per_clip])),
            "attitude_rmse_deg_std": float(np.std([r["attitude_rmse_deg"] for r in per_clip])),
            "altitude_rmse_mean": float(np.mean([r["altitude_rmse"] for r in per_clip])),
            "altitude_rmse_std": float(np.std([r["altitude_rmse"] for r in per_clip])),
            "position_note": "x/y dead-reckoned (no GPS); not a headline metric",
            "controls_mode": args.controls,
            "controls_note": (
                "zero controls + weaker nominal (gravity=0)"
                if args.controls == "zeros" and (args.gravity is None or args.gravity == 0.0)
                else (
                    "zero controls"
                    if args.controls == "zeros"
                    else f"exogenous hover thrust prior T={args.hover_thrust}"
                )
            ),
            "integrator_gravity": (
                float(args.gravity)
                if args.gravity is not None
                else (0.0 if args.controls == "zeros" else -9.81)
            ),
            "body_to_world_gt_velocity": not args.no_body_to_world,
            "protocol": "contiguous_window_dt_equals_1_over_fps",
            "baseline_v3_velocity_rmse": 1.29,
            "baseline_v3_attitude_rmse_deg": 33.6,
            "target_velocity_rmse": 1.29 * 0.5,
            "target_attitude_rmse_deg": 33.6 * 0.5,
            "sim_baseline_attitude_rmse_deg": 2.28,
            "sim_baseline_velocity_rmse": 0.075,
        }
        agg["velocity_reduction_pct"] = float(
            100.0 * (1.0 - agg["velocity_rmse_mean"] / 1.29)
        )
        agg["attitude_reduction_pct"] = float(
            100.0 * (1.0 - agg["attitude_rmse_deg_mean"] / 33.6)
        )
        agg["met_50pct_velocity"] = bool(agg["velocity_rmse_mean"] <= agg["target_velocity_rmse"])
        agg["met_50pct_attitude"] = bool(
            agg["attitude_rmse_deg_mean"] <= agg["target_attitude_rmse_deg"]
        )
        with open(out_dir / "real_data_metrics.json", "w") as f:
            json.dump({"per_clip": per_clip, "aggregate": agg}, f, indent=2)
        print(
            f"\n[real] aggregate: vel_rmse={agg['velocity_rmse_mean']:.4f} "
            f"({agg['velocity_reduction_pct']:+.1f}% vs v3)  "
            f"att_rmse={agg['attitude_rmse_deg_mean']:.4f} "
            f"({agg['attitude_reduction_pct']:+.1f}% vs v3)  "
            f"alt_rmse={agg['altitude_rmse_mean']:.4f}"
        )
        print(
            f"[real] 50% targets: vel≤{agg['target_velocity_rmse']:.3f} "
            f"[{'MET' if agg['met_50pct_velocity'] else 'MISS'}]  "
            f"att≤{agg['target_attitude_rmse_deg']:.2f} "
            f"[{'MET' if agg['met_50pct_attitude'] else 'MISS'}]"
        )

        n_show = min(4, len(all_pred_vel))
        fig, axes = plt.subplots(n_show, 2, figsize=(12, 3 * n_show), squeeze=False)
        for i in range(n_show):
            T = all_pred_vel[i].shape[0]
            xs = np.arange(1, T + 1)
            pred_vmag = np.sqrt((all_pred_vel[i] ** 2).sum(axis=1))
            gt_vmag = np.sqrt((all_gt_vel[i] ** 2).sum(axis=1))
            axes[i][0].plot(xs, gt_vmag, "k-", label="GT", linewidth=2)
            axes[i][0].plot(xs, pred_vmag, "b--", label="pred", linewidth=2)
            axes[i][0].set_title(f"clip {i}: velocity magnitude (m/s)")
            axes[i][0].legend()
            axes[i][0].grid(True, alpha=0.3)
            for j, name in enumerate(["yaw", "pitch", "roll"]):
                axes[i][1].plot(xs, all_gt_att[i][:, j], "k-", alpha=0.3)
                axes[i][1].plot(
                    xs,
                    all_pred_att[i][:, j],
                    ["b", "r", "g"][j] + "--",
                    label=f"pred {name}",
                )
            axes[i][1].set_title(f"clip {i}: attitude (deg)")
            axes[i][1].legend(fontsize=8)
            axes[i][1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "real_trajectories.png", dpi=120)
        print(f"[real] saved figures to {fig_dir}")
    print(f"[real] done. results in {out_dir}")


if __name__ == "__main__":
    main()
