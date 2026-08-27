"""Evaluation metrics for the AeroProber ablation study.

Computes the headline metrics from the charter:
- Position RMSE (m), overall and at specific horizons (e.g. 20-step).
- Attitude error (degrees): wrapped Euler-axis RMS **and** geodesic Log SO(3).
- Velocity RMSE (m/s) -- used for the real-data arm where position GT is absent.
- Long-horizon stability: does the rollout stay bounded to 30+ steps?
- Per-horizon error curves for the error-vs-horizon figure.

Legacy tables used wrapped Euler RMSE only. Prefer geodesic for SkyJEPA-comparable
reporting (see research/AUDIT.md V2 and COLLAB_MEMO.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from aerojepa_research.prober.integrator import _euler_ypr_to_rotation


def geodesic_attitude_error_deg(pred_att_deg: torch.Tensor, gt_att_deg: torch.Tensor) -> torch.Tensor:
    """Geodesic attitude error in degrees: angle of ``Log(R_gt^T R_pred)``.

    ``pred_att_deg`` / ``gt_att_deg``: (..., 3) yaw-pitch-roll in degrees.
    Returns (...,) geodesic angle in degrees.
    """
    R_pred = _euler_ypr_to_rotation(pred_att_deg)
    R_gt = _euler_ypr_to_rotation(gt_att_deg)
    R_err = torch.matmul(R_gt.transpose(-1, -2), R_pred)
    tr = R_err[..., 0, 0] + R_err[..., 1, 1] + R_err[..., 2, 2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    theta = torch.acos(cos_theta)
    return theta * (180.0 / np.pi)


@dataclass
class TrajectoryMetrics:
    """Metrics over a batch of predicted vs GT trajectories."""

    position_rmse: float  # meters
    velocity_rmse: float  # m/s
    attitude_rmse_deg: float  # degrees (wrapped Euler-axis RMS; legacy)
    attitude_rmse_geodesic_deg: float  # degrees (Log SO(3); preferred)
    angular_velocity_rmse: float  # deg/s
    per_horizon_pos_rmse: list[float]
    per_horizon_att_rmse: list[float]  # wrapped Euler (legacy)
    per_horizon_att_geodesic_rmse: list[float]
    n_samples: int


def compute_metrics(pred_stack: torch.Tensor, gt_stack: torch.Tensor) -> TrajectoryMetrics:
    """Compute trajectory metrics from (B, T, 12) predicted vs GT state stacks.

    Layout: [pos(3), vel(3), euler_att_deg(3), ang_vel(3)].
    """
    if not isinstance(pred_stack, torch.Tensor):
        pred_stack = torch.as_tensor(pred_stack)
    if not isinstance(gt_stack, torch.Tensor):
        gt_stack = torch.as_tensor(gt_stack)
    pred_t = pred_stack.detach().float()
    gt_t = gt_stack.detach().float()
    pred = pred_t.cpu().numpy()
    gt = gt_t.cpu().numpy()
    B, T, _ = pred.shape

    pos_err = np.sqrt(((pred[..., 0:3] - gt[..., 0:3]) ** 2).sum(axis=-1))
    vel_err = np.sqrt(((pred[..., 3:6] - gt[..., 3:6]) ** 2).sum(axis=-1))
    att_diff = ((pred[..., 6:9] - gt[..., 6:9] + 180.0) % 360.0) - 180.0
    att_err = np.sqrt((att_diff ** 2).sum(axis=-1))
    av_err = np.sqrt(((pred[..., 9:12] - gt[..., 9:12]) ** 2).sum(axis=-1))
    geo = geodesic_attitude_error_deg(pred_t[..., 6:9], gt_t[..., 6:9]).cpu().numpy()

    return TrajectoryMetrics(
        position_rmse=float(np.sqrt((pos_err ** 2).mean())),
        velocity_rmse=float(np.sqrt((vel_err ** 2).mean())),
        attitude_rmse_deg=float(np.sqrt((att_err ** 2).mean())),
        attitude_rmse_geodesic_deg=float(np.sqrt((geo ** 2).mean())),
        angular_velocity_rmse=float(np.sqrt((av_err ** 2).mean())),
        per_horizon_pos_rmse=[float(np.sqrt((pos_err[:, t] ** 2).mean())) for t in range(T)],
        per_horizon_att_rmse=[float(np.sqrt((att_err[:, t] ** 2).mean())) for t in range(T)],
        per_horizon_att_geodesic_rmse=[float(np.sqrt((geo[:, t] ** 2).mean())) for t in range(T)],
        n_samples=B,
    )


def metrics_to_dict(m: TrajectoryMetrics) -> dict:
    return {
        "position_rmse": m.position_rmse,
        "velocity_rmse": m.velocity_rmse,
        "attitude_rmse_deg": m.attitude_rmse_deg,
        "attitude_rmse_geodesic_deg": m.attitude_rmse_geodesic_deg,
        "angular_velocity_rmse": m.angular_velocity_rmse,
        "per_horizon_pos_rmse": m.per_horizon_pos_rmse,
        "per_horizon_att_rmse": m.per_horizon_att_rmse,
        "per_horizon_att_geodesic_rmse": m.per_horizon_att_geodesic_rmse,
        "n_samples": m.n_samples,
        "horizon_note": (
            "attitude_rmse_deg=wrapped Euler (legacy); prefer attitude_rmse_geodesic_deg"
        ),
    }


@torch.no_grad()
def evaluate_arm(
    extractor,
    model,
    loss_fn,
    loader,
    device,
    arm: str,
    max_loops: int | None = None,
) -> tuple[TrajectoryMetrics, list[np.ndarray], list[np.ndarray]]:
    """Run an arm over a loader and return aggregate metrics + raw trajectories."""
    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    for clips, _actions, states, controls in loader:
        rollout = extractor.extract(clips, controls, states, max_loops=max_loops)
        _loss, pred_stack = _compute_pred(arm, model, loss_fn, rollout)
        all_pred.append(pred_stack.detach().cpu().numpy())
        all_gt.append(rollout.gt_states.detach().cpu().numpy())

    pred_cat = np.concatenate(all_pred, axis=0)
    gt_cat = np.concatenate(all_gt, axis=0)
    metrics = compute_metrics(torch.from_numpy(pred_cat), torch.from_numpy(gt_cat))
    return metrics, all_pred, all_gt


def _compute_pred(arm, model, loss_fn, rollout):
    """Dispatch to the right forward call for the arm type."""
    if arm == "structured":
        return loss_fn(
            model, rollout.latents, rollout.controls, rollout.init_state, rollout.gt_states
        )
    if arm == "plain":
        return loss_fn(
            model, rollout.latents, rollout.controls, rollout.init_state, rollout.gt_states
        )
    if arm == "naive":
        return loss_fn(rollout.latents, rollout.controls, rollout.init_state, rollout.gt_states)
    raise ValueError(arm)
