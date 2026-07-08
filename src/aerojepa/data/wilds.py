from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from aerojepa.data.telemetry import ACTION_COLUMNS, derive_actions_from_raw

# The Wilds Drones (Parrot ANAFI) flight-log converter.
#
# HuggingFace dataset: imageomics/thewilds_drones
# Each mission folder contains ``video_files/*.MP4`` and ``metadata/*.json``
# Parrot flight logs with per-sample GPS, speeds, and Euler angles.

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".MP4")

# Parrot log fields we read (index in details_data row).
_LOG_FIELDS = (
    "time_ms",
    "speed_vx",
    "speed_vy",
    "speed_vz",
    "angle_phi",
    "angle_theta",
    "angle_psi",
    "altitude",
)


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "wilds converter needs opencv-python (`pip install opencv-python`)."
        ) from exc
    return cv2


def _video_meta(path: Path) -> tuple[int, float]:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 15.0
    cap.release()
    return max(0, frames), fps


def load_parrot_log(path: str | Path) -> np.ndarray:
    """Load a Parrot metadata JSON into ``(N, 8)`` raw-state rows.

    Columns: ``[t_sec, vx, vy, vz, altitude, yaw, pitch, roll]`` where angles
    are in radians (``angle_psi`` = yaw, ``angle_theta`` = pitch,
    ``angle_phi`` = roll).
    """
    data = json.loads(Path(path).read_text())
    headers = data["details_headers"]
    idx = {name: headers.index(name) for name in (
        "time", "speed_vx", "speed_vy", "speed_vz",
        "angle_phi", "angle_theta", "angle_psi", "altitude",
    )}

    rows = []
    for sample in data["details_data"]:
        t_sec = float(sample[idx["time"]]) / 1000.0
        rows.append([
            t_sec,
            float(sample[idx["speed_vx"]]),
            float(sample[idx["speed_vy"]]),
            float(sample[idx["speed_vz"]]),
            float(sample[idx["altitude"]]),
            float(sample[idx["angle_psi"]]),    # yaw
            float(sample[idx["angle_theta"]]),  # pitch
            float(sample[idx["angle_phi"]]),    # roll
        ])
    return np.asarray(rows, dtype=np.float32)


def resample_log_to_frames(log: np.ndarray, num_frames: int, fps: float) -> np.ndarray:
    """Linearly resample a flight log onto the video frame timeline.

    Returns ``(num_frames, 8)`` with the same columns as :func:`load_parrot_log`.
    """
    if num_frames <= 0:
        return np.zeros((0, 8), dtype=np.float32)
    if log.size == 0:
        return np.zeros((num_frames, 8), dtype=np.float32)

    frame_times = np.arange(num_frames, dtype=np.float32) / max(fps, 1e-6)
    log_times = log[:, 0]
    out = np.zeros((num_frames, 8), dtype=np.float32)
    for col in range(8):
        out[:, col] = np.interp(frame_times, log_times, log[:, col])
    return out


def parrot_log_to_actions(log: np.ndarray, num_frames: int, fps: float) -> np.ndarray:
    """Resample a Parrot log and derive ACTION_COLUMNS for ``num_frames``."""
    resampled = resample_log_to_frames(log, num_frames, fps)
    # Map to derive_actions_from_raw layout: [t, vgx, vgy, vgz, height, yaw, pitch, roll]
    raw = np.zeros((num_frames, 8), dtype=np.float32)
    raw[:, 0] = resampled[:, 0]
    raw[:, 1:4] = resampled[:, 1:4]
    raw[:, 4] = resampled[:, 4]                          # altitude as height proxy
    raw[:, 5:8] = np.degrees(resampled[:, 5:8])          # radians -> degrees for wrap
    return derive_actions_from_raw(raw)


def _write_csv(path: Path, actions: np.ndarray) -> None:
    with path.open("w") as f:
        f.write(",".join(ACTION_COLUMNS) + "\n")
        for row in actions:
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def _mission_dirs(raw_dir: Path) -> list[Path]:
    return sorted(
        p for p in raw_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".") and re.match(r"\d{8}_", p.name)
    )


def _pair_videos_jsons(mission_dir: Path) -> list[tuple[Path, Path | None]]:
    video_dir = mission_dir / "video_files"
    if not video_dir.exists():
        video_dir = mission_dir / "videos"
    meta_dir = mission_dir / "metadata"
    videos = sorted(
        p for p in video_dir.glob("*") if p.suffix in VIDEO_SUFFIXES or p.suffix.lower() in VIDEO_SUFFIXES
    )
    jsons = sorted(meta_dir.glob("*.json")) if meta_dir.exists() else []
    pairs: list[tuple[Path, Path | None]] = []
    for i, video in enumerate(videos):
        telem = jsons[i] if i < len(jsons) else (jsons[0] if jsons else None)
        pairs.append((video, telem))
    return pairs


def convert_wilds(
    raw_dir: str | Path,
    out_dir: str | Path,
    *,
    link_videos: bool = True,
) -> list[Path]:
    """Convert The Wilds Drones raw tree into AeroJEPA ``data/flights/`` layout.

    Writes ``<name>.mp4`` + sibling ``<name>.csv`` under ``out_dir``. Videos are
    symlinked by default (``link_videos=True``) to avoid duplicating gigabytes.
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
                actions = parrot_log_to_actions(log, num_frames, fps)
                _write_csv(out_video.with_suffix(".csv"), actions)
            written.append(out_video)

    return written


def discover_wilds_videos(raw_dir: str | Path) -> list[Path]:
    """Return all video files under a Wilds raw tree."""
    raw_dir = Path(raw_dir)
    out: list[Path] = []
    for mission in _mission_dirs(raw_dir):
        for video, _ in _pair_videos_jsons(mission):
            if video.exists() and video.stat().st_size > 0:
                out.append(video)
    return out
