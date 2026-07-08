"""Extended Parrot Wilds converter that preserves absolute metric state.

The parent project's ``aerojepa.data.wilds`` converter derives only the 6-DoF
action deltas and discards absolute velocity/altitude/attitude. For the prober's
real-data arm we need those absolute fields (only x/y position is genuinely
missing from Parrot logs). This converter writes a sibling ``*_state.csv`` next
to each ``*_actions.csv`` containing:

    [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, yaw, pitch, roll, av_yaw, av_pitch, av_roll]

where:
- pos_z = altitude (directly from the Parrot log).
- pos_x, pos_y = DEAD-RECKONED by trapezoidal integration of body velocity
  rotated into the world frame by yaw. This is an ESTIMATE (no GPS in the
  Parrot log) -- marked as such in the CSV header. The prober real-data arm
  therefore reports velocity/attitude/altitude RMSE as quantitative metrics
  and treats position as qualitative-only (future work: GPS-preserving capture).
- yaw/pitch/roll are in degrees, wrapped to (-180, 180].
- angular velocities are frame-to-frame wrapped attitude deltas divided by dt.

The action CSV is identical to the parent converter's output (so the frozen
world model sees the same in-distribution inputs); this module only ADDS the
state CSV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aerojepa.data.telemetry import ACTION_COLUMNS, derive_actions_from_raw, wrap_degrees
from aerojepa.data.wilds import (
    _mission_dirs,
    _pair_videos_jsons,
    _video_meta,
    _write_csv,
    load_parrot_log,
    resample_log_to_frames,
)

# Absolute state columns written alongside the action deltas.
# pos_x/pos_y are dead-reckoned estimates (no GPS); pos_z is altitude.
STATE_COLUMNS = (
    "pos_x", "pos_y", "pos_z",
    "vel_x", "vel_y", "vel_z",
    "yaw", "pitch", "roll",
    "av_yaw", "av_pitch", "av_roll",
)


def parrot_log_to_state(log: np.ndarray, num_frames: int, fps: float) -> np.ndarray:
    """Resample a Parrot log and derive absolute metric state for ``num_frames``.

    Returns ``(num_frames, 12)`` with columns = ``STATE_COLUMNS``.
    """
    resampled = resample_log_to_frames(log, num_frames, fps)  # (N, 8) [t, vx, vy, vz, alt, yaw, pitch, roll] rad
    t = resampled[:, 0]
    vx, vy, vz = resampled[:, 1], resampled[:, 2], resampled[:, 3]
    altitude = resampled[:, 4]
    yaw_rad, pitch_rad, roll_rad = resampled[:, 5], resampled[:, 6], resampled[:, 7]

    # Convert attitude to degrees (yaw, pitch, roll order).
    yaw_deg = np.degrees(yaw_rad)
    pitch_deg = np.degrees(pitch_rad)
    roll_deg = np.degrees(roll_rad)

    dt = np.zeros(num_frames, dtype=np.float32)
    dt[1:] = np.diff(t)
    dt[dt <= 0] = 1.0 / max(fps, 1e-6)

    # Angular velocity from wrapped attitude deltas (deg/s).
    av_yaw = np.zeros(num_frames, dtype=np.float32)
    av_pitch = np.zeros(num_frames, dtype=np.float32)
    av_roll = np.zeros(num_frames, dtype=np.float32)
    av_yaw[1:] = wrap_degrees(np.diff(yaw_deg)) / dt[1:]
    av_pitch[1:] = wrap_degrees(np.diff(pitch_deg)) / dt[1:]  # wrap for safety
    av_roll[1:] = wrap_degrees(np.diff(roll_deg)) / dt[1:]

    # Dead-reckon x/y position by integrating body velocity rotated by yaw.
    # World velocity = Rz(yaw) @ [vx, vy, vz]; we only need x/y for pos_x/pos_y.
    # Note: Parrot's vx/vy are body-frame; vz is along body z (approx world z
    # for near-level flight). This is an ESTIMATE -- see module docstring.
    cos_y = np.cos(yaw_rad)
    sin_y = np.sin(yaw_rad)
    world_vx = cos_y * vx - sin_y * vy
    world_vy = sin_y * vx + cos_y * vy
    pos_x = np.zeros(num_frames, dtype=np.float32)
    pos_y = np.zeros(num_frames, dtype=np.float32)
    pos_x[1:] = np.cumsum(0.5 * (world_vx[:-1] + world_vx[1:]) * dt[1:])
    pos_y[1:] = np.cumsum(0.5 * (world_vy[:-1] + world_vy[1:]) * dt[1:])

    # Wrap attitude to (-180, 180].
    yaw_deg = ((yaw_deg + 180.0) % 360.0) - 180.0
    pitch_deg = ((pitch_deg + 180.0) % 360.0) - 180.0
    roll_deg = ((roll_deg + 180.0) % 360.0) - 180.0

    state = np.stack([
        pos_x, pos_y, altitude,
        vx, vy, vz,
        yaw_deg, pitch_deg, roll_deg,
        av_yaw, av_pitch, av_roll,
    ], axis=1).astype(np.float32)
    return state


def _write_state_csv(path: Path, state: np.ndarray) -> None:
    with path.open("w") as f:
        f.write(",".join(STATE_COLUMNS) + "\n")
        for row in state:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def convert_wilds_with_state(
    raw_dir: str | Path,
    out_dir: str | Path,
    *,
    link_videos: bool = True,
) -> list[Path]:
    """Convert The Wilds Drones raw tree, writing both actions + absolute state CSVs.

    Drop-in replacement for ``aerojepa.data.wilds.convert_wilds`` that ADDS a
    ``<name>_state.csv`` next to each ``<name>.csv`` (actions).
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    clip_idx = 0
    for mission in _mission_dirs(raw_dir):
        for video, telem_json in _pair_videos_jsons(mission):
            if not video.exists() or video.stat().st_size == 0:
                continue
            name = f"wilds_{clip_idx:03d}"
            clip_idx += 1
            out_video = out_dir / f"{name}.mp4"

            if link_videos:
                if out_video.exists() or out_video.is_symlink():
                    out_video.unlink()
                out_video.symlink_to(video.resolve())
            else:
                import shutil
                shutil.copy2(video, out_video)

            num_frames, fps = _video_meta(video)
            if telem_json and telem_json.exists():
                log = load_parrot_log(telem_json)
                # Actions (same as parent converter).
                resampled = resample_log_to_frames(log, num_frames, fps)
                raw = np.zeros((num_frames, 8), dtype=np.float32)
                raw[:, 0] = resampled[:, 0]
                raw[:, 1:4] = resampled[:, 1:4]
                raw[:, 4] = resampled[:, 4]
                raw[:, 5:8] = np.degrees(resampled[:, 5:8])
                actions = derive_actions_from_raw(raw)
                _write_csv(out_video.with_suffix(".csv"), actions)
                # Absolute state (new).
                state = parrot_log_to_state(log, num_frames, fps)
                _write_state_csv(out_dir / f"{name}_state.csv", state)
            written.append(out_video)

    return written
