from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# The 6-DoF action/telemetry convention used throughout AeroJEPA.
# Each row describes the motion that produced one frame.
ACTION_COLUMNS = ("dx", "dy", "d_altitude", "d_yaw", "d_pitch", "d_roll")

# Rich raw flight-log schema written alongside captures (provenance). Training
# never reads this directly -- it reads the derived ACTION_COLUMNS CSV -- but the
# raw log lets us re-derive actions later or debug a flight. ``t`` is seconds
# since capture start.
RAW_STATE_COLUMNS = (
    "t", "vgx", "vgy", "vgz", "height", "yaw", "pitch", "roll", "bat", "tof", "baro"
)


def wrap_degrees(delta: np.ndarray) -> np.ndarray:
    """Wrap angle differences to (-180, 180] so a 359->1 step reads as +2, not -358."""
    return (delta + 180.0) % 360.0 - 180.0


def derive_actions_from_raw(raw: np.ndarray) -> np.ndarray:
    """Convert a raw flight log into the 6-DoF ACTION_COLUMNS convention.

    ``raw`` is ``(N, >=8)`` with the first eight columns
    ``[t, vgx, vgy, vgz, height, yaw, pitch, roll]`` (extra columns ignored).

    The three **linear** channels use the drone's reported body velocities
    (``vgx, vgy, vgz``) -- a direct proxy for "how far it moved this frame". The
    three **angular** channels use wrapped frame-to-frame *deltas* of the absolute
    attitude angles, because that is the motion that actually happened between
    frames. The first row's angular deltas are zero (no previous frame).

    Returns ``(N, 6)`` float32. This fixes a subtle bug in early captures that
    stored *absolute* yaw/pitch/roll in columns meant for deltas.
    """
    raw = np.atleast_2d(np.asarray(raw, dtype=np.float32))
    n = raw.shape[0]
    actions = np.zeros((n, len(ACTION_COLUMNS)), dtype=np.float32)
    if n == 0:
        return actions

    actions[:, 0] = raw[:, 1]  # dx  <- vgx
    actions[:, 1] = raw[:, 2]  # dy  <- vgy
    actions[:, 2] = raw[:, 3]  # d_altitude <- vgz

    yaw, pitch, roll = raw[:, 5], raw[:, 6], raw[:, 7]
    actions[1:, 3] = wrap_degrees(np.diff(yaw))
    actions[1:, 4] = np.diff(pitch)
    actions[1:, 5] = np.diff(roll)
    return actions


def normalize_actions(actions: torch.Tensor, scale: float = 0.1) -> torch.Tensor:
    """Scale raw pose deltas into a roughly unit range for the action encoder.

    Real telemetry arrives in mixed units (metres, radians); a single fixed
    scale keeps the action embedding well-conditioned without per-dataset
    hand-tuning. ``tanh`` bounds outliers so one wild reading cannot dominate.
    """
    return torch.tanh(actions / scale)


def load_telemetry_all(path: str | Path) -> torch.Tensor | None:
    """Load the full telemetry table for a clip, or ``None`` if it is missing.

    Returns an ``(N, 6)`` tensor of per-frame pose deltas (columns =
    ``ACTION_COLUMNS``). Extra columns are ignored so richer flight logs can be
    dropped in unchanged. ``None`` signals "no telemetry for this clip" so the
    caller can fall back to zeros (the masked objective still trains without it).
    """
    path = Path(path)
    if not path.exists():
        return None
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2).astype(np.float32)
    if rows.size == 0:
        return None
    return torch.from_numpy(rows[:, : len(ACTION_COLUMNS)])


def gather_telemetry(
    path: str | Path, indices: list[int], num_frames: int
) -> torch.Tensor:
    """Select the telemetry rows aligned with a set of sampled frame indices.

    Frames and telemetry share a timeline, so once the dataset has chosen which
    frame indices make up a clip (uniform sampling or a sliding window), the
    matching pose deltas are gathered with the same indices. Missing files, out
    of range indices, or short logs are all handled by returning zeros for the
    affected rows, giving a well-formed ``(num_frames, 6)`` tensor every time.
    """
    table = load_telemetry_all(path)
    out = torch.zeros(num_frames, len(ACTION_COLUMNS))
    if table is None:
        return out
    n = table.shape[0]
    for slot, idx in enumerate(indices[:num_frames]):
        if 0 <= idx < n:
            out[slot] = table[idx]
        elif n > 0:
            out[slot] = table[min(idx, n - 1)]  # clamp padded/overshoot indices
    return out


def load_telemetry_csv(path: str | Path, num_frames: int) -> torch.Tensor:
    """Load a telemetry CSV (one row per frame, columns = ``ACTION_COLUMNS``).

    Missing or short files are padded/truncated to ``num_frames`` so a clip can
    always be assembled even when logging dropped a few samples. Kept for
    backward compatibility; new code should prefer :func:`gather_telemetry`,
    which keeps telemetry aligned with sampled frame indices.
    """
    path = Path(path)
    if not path.exists():
        return torch.zeros(num_frames, len(ACTION_COLUMNS))

    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2).astype(np.float32)
    actions = torch.from_numpy(rows[:, : len(ACTION_COLUMNS)])
    if actions.shape[0] >= num_frames:
        return actions[:num_frames]
    pad = torch.zeros(num_frames - actions.shape[0], len(ACTION_COLUMNS))
    return torch.cat([actions, pad], dim=0)
