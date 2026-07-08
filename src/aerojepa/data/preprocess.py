from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from aerojepa.data.telemetry import (
    ACTION_COLUMNS,
    derive_actions_from_raw,
    load_telemetry_all,
)

# Turn *any* footage into training-ready clips.
#
# Real video arrives in many shapes: Tello captures, phone clips, downloaded
# drone footage, mixed frame rates and resolutions. This module standardizes them
# into the layout ``VideoClipDataset`` expects -- consistent codec, frame rate,
# and optional square framing -- and keeps telemetry aligned when it exists. It
# also probes a folder ("dataset doctor") so you can see exactly what you have
# before training.
#
# OpenCV-only (no ffmpeg), so it runs out of the box on a Mac Studio.

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "preprocess needs opencv-python. Install with `pip install opencv-python`."
        ) from exc
    return cv2


@dataclass
class ClipInfo:
    path: Path
    frames: int
    fps: float
    duration_s: float
    width: int
    height: int
    telemetry: str = "none"       # "actions" | "raw" | "none"
    telemetry_rows: int = 0
    issues: list[str] = field(default_factory=list)


def probe_clip(path: str | Path) -> ClipInfo:
    """Report a clip's frame count, fps, duration, resolution, and telemetry state.

    Flags common problems (empty/short clip, telemetry-row vs frame-count
    mismatch) in ``issues`` so a bad clip is caught before it wastes a training run.
    """
    cv2 = _require_cv2()
    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = frames / fps if fps > 0 else 0.0
    info = ClipInfo(path, frames, fps, duration, width, height)

    action_csv = path.with_suffix(".csv")
    raw_csv = path.with_suffix(".raw.csv")
    table = None
    if action_csv.exists():
        info.telemetry = "actions"
        table = load_telemetry_all(action_csv)
    elif raw_csv.exists():
        info.telemetry = "raw"
        table = load_telemetry_all(raw_csv)
    if table is not None:
        info.telemetry_rows = int(table.shape[0])

    if frames <= 0:
        info.issues.append("no frames (unreadable or empty)")
    if 0 < frames < 8:
        info.issues.append(f"very short ({frames} frames)")
    if info.telemetry != "none" and info.telemetry_rows and frames:
        drift = abs(info.telemetry_rows - frames) / frames
        if drift > 0.1:
            info.issues.append(
                f"telemetry rows ({info.telemetry_rows}) != frames ({frames})"
            )
    return info


def probe_directory(data_dir: str | Path) -> list[ClipInfo]:
    data_dir = Path(data_dir)
    paths = sorted(p for p in data_dir.glob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    return [probe_clip(p) for p in paths]


def _center_square(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    return frame[y0 : y0 + side, x0 : x0 + side]


def _select_indices(src_frames: int, src_fps: float, target_fps: int, max_seconds: float | None) -> list[int]:
    """Pick source frame indices to resample a clip to ``target_fps``.

    Uses even temporal spacing (stride) so motion stays smooth. Caps the output
    length at ``max_seconds`` of the *target* timeline when given.
    """
    if src_frames <= 0:
        return []
    src_fps = src_fps if src_fps > 0 else float(target_fps)
    duration = src_frames / src_fps
    if max_seconds is not None:
        duration = min(duration, max_seconds)
    n_out = max(1, int(round(duration * target_fps)))
    idx = np.linspace(0, src_frames - 1, n_out).round().astype(int)
    return idx.tolist()


def _load_source_actions(src_path: Path) -> np.ndarray | None:
    """Return an (N, 6) action table for a source clip, deriving from raw if needed."""
    action_csv = src_path.with_suffix(".csv")
    raw_csv = src_path.with_suffix(".raw.csv")
    if action_csv.exists():
        table = load_telemetry_all(action_csv)
        return None if table is None else table.numpy()
    if raw_csv.exists():
        raw = np.loadtxt(raw_csv, delimiter=",", skiprows=1, ndmin=2).astype(np.float32)
        return derive_actions_from_raw(raw) if raw.size else None
    return None


def standardize_clip(
    src_path: str | Path,
    out_path: str | Path,
    target_fps: int = 15,
    max_seconds: float | None = None,
    square: bool = False,
    resize: int | None = None,
) -> ClipInfo:
    """Re-encode one clip to a standard fps/codec, keeping telemetry aligned.

    Writes an mp4 to ``out_path`` and, if the source has telemetry, a sibling
    ``.csv`` in ACTION_COLUMNS format resampled to the same frames. Returns a
    :class:`ClipInfo` for the *output* clip.
    """
    cv2 = _require_cv2()
    src_path = Path(src_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src_path))
    src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = float(cap.get(cv2.CAP_PROP_FPS)) or float(target_fps)
    indices = _select_indices(src_frames, src_fps, target_fps, max_seconds)
    index_set = set(indices)

    # Single sequential pass; keep frames whose index was selected (indices are
    # sorted and may repeat when upsampling, so we cache the last kept frame).
    kept: dict[int, np.ndarray] = {}
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in index_set:
            kept[i] = frame
        i += 1
    cap.release()

    ordered = [kept[j] for j in indices if j in kept]
    if not ordered:
        raise RuntimeError(f"Could not read frames from {src_path}")

    if square:
        ordered = [_center_square(f) for f in ordered]
    if resize:
        ordered = [cv2.resize(f, (resize, resize)) for f in ordered]

    h, w = ordered[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), target_fps, (w, h))
    for frame in ordered:
        writer.write(frame)
    writer.release()

    # Align telemetry to the same selected indices.
    actions = _load_source_actions(src_path)
    if actions is not None and len(actions):
        sel = [min(j, len(actions) - 1) for j in indices if j in kept]
        aligned = actions[sel]
        out_csv = out_path.with_suffix(".csv")
        with out_csv.open("w") as f:
            f.write(",".join(ACTION_COLUMNS) + "\n")
            for row in aligned:
                f.write(",".join(f"{v:.6f}" for v in row[: len(ACTION_COLUMNS)]) + "\n")

    return probe_clip(out_path)


@dataclass
class PreprocessConfig:
    input_dir: str | Path
    output_dir: str | Path = "data/flights"
    target_fps: int = 15
    max_seconds: float | None = None
    square: bool = False
    resize: int | None = None
    prefix: str = "clip"


def preprocess_directory(config: PreprocessConfig) -> list[ClipInfo]:
    """Standardize every video in ``input_dir`` into ``output_dir``.

    Output clips are renamed ``<prefix>_000.mp4``, ``<prefix>_001.mp4``, ... so a
    training folder stays tidy regardless of messy source names.
    """
    in_dir = Path(config.input_dir)
    out_dir = Path(config.output_dir)
    sources = sorted(p for p in in_dir.glob("*") if p.suffix.lower() in VIDEO_SUFFIXES)
    if not sources:
        raise FileNotFoundError(f"No videos found in {in_dir}")

    results: list[ClipInfo] = []
    for i, src in enumerate(sources):
        out_path = out_dir / f"{config.prefix}_{i:03d}.mp4"
        info = standardize_clip(
            src,
            out_path,
            target_fps=config.target_fps,
            max_seconds=config.max_seconds,
            square=config.square,
            resize=config.resize,
        )
        results.append(info)
        print(f"  {src.name} -> {out_path.name}  ({info.frames}f @ {config.target_fps}fps, tel={info.telemetry})")
    return results
