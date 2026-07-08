from __future__ import annotations

import numpy as np
import pytest

from aerojepa.data.telemetry import (
    ACTION_COLUMNS,
    RAW_STATE_COLUMNS,
    derive_actions_from_raw,
    load_telemetry_all,
    wrap_degrees,
)

cv2 = pytest.importorskip("cv2", reason="opencv-python required for preprocess tests")

from aerojepa.data.preprocess import (
    PreprocessConfig,
    preprocess_directory,
    probe_clip,
    standardize_clip,
)


def _write_video(path, num_frames: int, fps: int = 30, size: int = 48) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (size, size))
    assert writer.isOpened()
    for i in range(num_frames):
        shade = int(255 * i / max(1, num_frames - 1))
        writer.write(np.full((size, size, 3), shade, dtype=np.uint8))
    writer.release()


# ---- telemetry derivation -------------------------------------------------


def test_wrap_degrees_handles_wraparound() -> None:
    # 359 -> 1 should read as +2, not -358.
    assert wrap_degrees(np.array([-358.0]))[0] == pytest.approx(2.0)
    assert wrap_degrees(np.array([358.0]))[0] == pytest.approx(-2.0)


def test_derive_actions_from_raw_shapes_and_semantics() -> None:
    n = 5
    raw = np.zeros((n, len(RAW_STATE_COLUMNS)), dtype=np.float32)
    raw[:, 1] = 2.0          # vgx -> dx
    raw[:, 2] = -1.0         # vgy -> dy
    raw[:, 3] = 0.5          # vgz -> d_altitude
    raw[:, 5] = np.array([0, 10, 20, 30, 40])  # yaw absolute
    actions = derive_actions_from_raw(raw)

    assert actions.shape == (n, len(ACTION_COLUMNS))
    # Linear channels are the velocities directly.
    assert np.allclose(actions[:, 0], 2.0)
    assert np.allclose(actions[:, 1], -1.0)
    assert np.allclose(actions[:, 2], 0.5)
    # Angular channel is a per-frame delta; first row has no previous frame.
    assert actions[0, 3] == pytest.approx(0.0)
    assert np.allclose(actions[1:, 3], 10.0)


def test_derive_actions_empty() -> None:
    assert derive_actions_from_raw(np.zeros((0, 8))).shape == (0, len(ACTION_COLUMNS))


# ---- probe / standardize --------------------------------------------------


def test_probe_reports_metadata_and_issues(tmp_path) -> None:
    _write_video(tmp_path / "a.mp4", num_frames=30, fps=30)
    info = probe_clip(tmp_path / "a.mp4")
    assert info.frames == 30
    assert info.telemetry == "none"

    _write_video(tmp_path / "short.mp4", num_frames=3, fps=30)
    short = probe_clip(tmp_path / "short.mp4")
    assert any("short" in msg for msg in short.issues)


def test_standardize_resamples_fps_and_aligns_telemetry(tmp_path) -> None:
    src = tmp_path / "src.mp4"
    _write_video(src, num_frames=30, fps=30)  # 1.0s of footage
    # Raw telemetry with a steadily increasing yaw so alignment is checkable.
    raw = np.zeros((30, len(RAW_STATE_COLUMNS)), dtype=np.float32)
    raw[:, 1] = 1.0
    raw[:, 5] = np.arange(30)
    np.savetxt(
        src.with_suffix(".raw.csv"), raw, delimiter=",",
        header=",".join(RAW_STATE_COLUMNS), comments="",
    )

    out = tmp_path / "out.mp4"
    info = standardize_clip(src, out, target_fps=15)
    # 1.0s @ 15fps -> ~15 frames.
    assert 14 <= info.frames <= 16
    assert info.telemetry == "actions"

    table = load_telemetry_all(out.with_suffix(".csv"))
    assert table is not None
    assert table.shape[1] == len(ACTION_COLUMNS)
    assert table.shape[0] == info.frames
    assert np.allclose(table[:, 0].numpy(), 1.0)  # dx preserved from vgx


def test_standardize_square_crop(tmp_path) -> None:
    src = tmp_path / "wide.mp4"
    writer = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 32))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.zeros((32, 64, 3), dtype=np.uint8))
    writer.release()

    out = tmp_path / "sq.mp4"
    info = standardize_clip(src, out, target_fps=15, square=True)
    assert info.width == info.height == 32


def test_preprocess_directory_renames_and_writes(tmp_path) -> None:
    in_dir = tmp_path / "raw"
    in_dir.mkdir()
    _write_video(in_dir / "messy name (1).mp4", num_frames=20, fps=20)
    _write_video(in_dir / "another.mov", num_frames=20, fps=20)
    out_dir = tmp_path / "flights"

    results = preprocess_directory(
        PreprocessConfig(input_dir=in_dir, output_dir=out_dir, target_fps=10, prefix="clip")
    )
    assert len(results) == 2
    assert (out_dir / "clip_000.mp4").exists()
    assert (out_dir / "clip_001.mp4").exists()
