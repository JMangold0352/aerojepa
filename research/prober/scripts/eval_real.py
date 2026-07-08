"""Real-data evaluation on Parrot Wilds (partial-quantitative).

Loads a trained structured prober + the frozen real_finetune_fast checkpoint,
runs the prober on real Wilds clips, and reports quantitative metrics where
ground truth exists: velocity RMSE, attitude RMSE, altitude RMSE. Position x/y
is dead-reckoned (no GPS) so it is reported qualitatively only -- flagged as
future work.

Outputs:
- real_data_metrics.json : per-clip + aggregate metrics.
- figures/real_trajectories.png : predicted vs GT velocity + attitude over time.

Usage:
    python research/prober/scripts/eval_real.py \
        --prober research/prober/results/prober_regular/best.pt \
        --checkpoint checkpoints/real_finetune_fast/latest.pt \
        --data-dir data/flights_with_state \
        --num-clips 16

NOTE: runs OUTSIDE the Cursor sandbox (needs video decoding + PyBullet-free torch).
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

from aerojepa_research.prober.integrator import KinematicIntegrator, MetricState
from aerojepa_research.prober.metrics import compute_metrics, metrics_to_dict
from aerojepa_research.prober.prober import Prober
from aerojepa_research.prober.rollout import FrozenRolloutExtractor
from aerojepa_research.prober.wilds_state import STATE_COLUMNS


def load_clip_with_state(video_path: Path, state_csv: Path, num_frames: int, img_size: int, device):
    """Load one clip (uniformly sampled) + its action + state CSVs."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        cap.release()
        return None
    # Uniform sample num_frames indices across the whole video.
    idxs = np.linspace(0, total - 1, num_frames).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame[:, :, :3])
    cap.release()
    clips = torch.from_numpy(np.stack(frames)).float() / 255.0  # (T, H, W, 3)
    clips = clips.permute(0, 3, 1, 2)  # (T, 3, H, W)

    # Load state CSV; gather the rows matching our sampled frame indices.
    state_all = np.loadtxt(state_csv, delimiter=",", skiprows=1, ndmin=2).astype(np.float32)
    # Map video frame indices to state rows (state CSV has one row per video frame).
    state_rows = np.zeros((num_frames, 12), dtype=np.float32)
    for slot, fi in enumerate(idxs):
        state_rows[slot] = state_all[min(fi, len(state_all) - 1)]
    metric_states = torch.from_numpy(state_rows)

    # Actions: derive from state (velocity + attitude deltas), matching the
    # data_pyflyt.states_to_actions convention.
    from aerojepa_research.prober.data_pyflyt import states_to_actions
    actions = torch.from_numpy(states_to_actions(state_rows))

    return clips.to(device), actions.to(device), metric_states.to(device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prober", required=True, help="path to trained prober best.pt")
    parser.add_argument("--checkpoint", default="checkpoints/real_finetune_fast/latest.pt")
    parser.add_argument("--data-dir", default="data/flights_with_state")
    parser.add_argument("--num-clips", type=int, default=16)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--context-frames", type=int, default=4)
    parser.add_argument("--output-dir", default="research/prober/results/real_data")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[real] device={device} checkpoint={args.checkpoint}")

    # Frozen extractor on the real-finetuned checkpoint.
    extractor = FrozenRolloutExtractor(
        args.checkpoint, device=device, context_frames=args.context_frames,
    )
    img_size = extractor.cfg["data"]["img_size"]
    print(f"[real] img_size={img_size} looped={extractor.is_looped()}")

    # Trained prober.
    prober = Prober().to(device)
    ckpt = torch.load(args.prober, map_location=device, weights_only=False)
    prober.load_state_dict(ckpt["model"])
    prober.eval()
    integ = KinematicIntegrator(dt=1.0 / 15.0, gravity=-9.81).to(device)

    # Discover clips that have BOTH a video and a state CSV.
    data_dir = Path(args.data_dir)
    clips = sorted(data_dir.glob("wilds_*.mp4"))[: args.num_clips]
    print(f"[real] found {len(clips)} clips")

    per_clip = []
    all_pred_vel, all_gt_vel = [], []
    all_pred_att, all_gt_att = [], []
    for vi, vp in enumerate(clips):
        state_csv = data_dir / (vp.stem + "_state.csv")
        if not state_csv.exists():
            print(f"[real] clip {vp.stem}: no state csv, skipping")
            continue
        batch = load_clip_with_state(vp, state_csv, args.num_frames, img_size, device)
        if batch is None:
            continue
        clips_t, actions_t, states_t = batch
        clips_b = clips_t.unsqueeze(0)
        actions_b = actions_t.unsqueeze(0)
        states_b = states_t.unsqueeze(0)

        with torch.no_grad():
            rollout = extractor.extract(clips_b, actions_b, states_b, max_loops=1)
            res_lin, res_ang = prober(rollout.latents, rollout.actions)
            pred_traj = integ.rollout(rollout.init_state, rollout.actions, res_lin, res_ang)
            pred_stack = pred_traj.stack()

        m = compute_metrics(pred_stack, rollout.gt_states)
        per_clip.append({"clip": vp.stem, **metrics_to_dict(m)})
        all_pred_vel.append(pred_stack[0, :, 3:6].cpu().numpy())
        all_gt_vel.append(rollout.gt_states[0, :, 3:6].cpu().numpy())
        all_pred_att.append(pred_stack[0, :, 6:9].cpu().numpy())
        all_gt_att.append(rollout.gt_states[0, :, 6:9].cpu().numpy())
        print(
            f"[real] clip {vp.stem}: vel_rmse={m.velocity_rmse:.4f} "
            f"att_rmse={m.attitude_rmse_deg:.4f} "
            f"alt_rmse={float(np.sqrt(((pred_stack[0,:,2].cpu().numpy()-rollout.gt_states[0,:,2].cpu().numpy())**2).mean())):.4f}"
        )

    # Aggregate.
    if per_clip:
        agg = {
            "n_clips": len(per_clip),
            "velocity_rmse_mean": float(np.mean([r["velocity_rmse"] for r in per_clip])),
            "velocity_rmse_std": float(np.std([r["velocity_rmse"] for r in per_clip])),
            "attitude_rmse_deg_mean": float(np.mean([r["attitude_rmse_deg"] for r in per_clip])),
            "attitude_rmse_deg_std": float(np.std([r["attitude_rmse_deg"] for r in per_clip])),
            "position_note": "x/y dead-reckoned (no GPS); altitude RMSE quantitative only",
        }
        with open(out_dir / "real_data_metrics.json", "w") as f:
            json.dump({"per_clip": per_clip, "aggregate": agg}, f, indent=2)
        print(f"\n[real] aggregate: vel_rmse={agg['velocity_rmse_mean']:.4f} "
              f"att_rmse={agg['attitude_rmse_deg_mean']:.4f}")

        # Trajectory figure: velocity + attitude for first few clips.
        n_show = min(4, len(all_pred_vel))
        fig, axes = plt.subplots(n_show, 2, figsize=(12, 3 * n_show), squeeze=False)
        for i in range(n_show):
            T = all_pred_vel[i].shape[0]
            xs = np.arange(1, T + 1)
            # Velocity magnitude.
            pred_vmag = np.sqrt((all_pred_vel[i] ** 2).sum(axis=1))
            gt_vmag = np.sqrt((all_gt_vel[i] ** 2).sum(axis=1))
            axes[i][0].plot(xs, gt_vmag, "k-", label="GT", linewidth=2)
            axes[i][0].plot(xs, pred_vmag, "b--", label="pred", linewidth=2)
            axes[i][0].set_title(f"clip {i}: velocity magnitude (m/s)")
            axes[i][0].legend(); axes[i][0].grid(True, alpha=0.3)
            # Attitude (yaw/pitch/roll).
            for j, name in enumerate(["yaw", "pitch", "roll"]):
                axes[i][1].plot(xs, all_gt_att[i][:, j], f"k-", alpha=0.3)
                axes[i][1].plot(xs, all_pred_att[i][:, j], ["b", "r", "g"][j] + "--", label=f"pred {name}")
            axes[i][1].set_title(f"clip {i}: attitude (deg)")
            axes[i][1].legend(fontsize=8); axes[i][1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fig_dir / "real_trajectories.png", dpi=120)
        print(f"[real] saved figures to {fig_dir}")
    print(f"[real] done. results in {out_dir}")


if __name__ == "__main__":
    main()
