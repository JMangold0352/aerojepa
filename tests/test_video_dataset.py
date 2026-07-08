from __future__ import annotations

import numpy as np
import pytest
import torch

cv2 = pytest.importorskip("cv2", reason="opencv-python required for video dataset tests")

from aerojepa.data.telemetry import ACTION_COLUMNS, gather_telemetry, load_telemetry_all
from aerojepa.data.video_dataset import VideoClipDataset, discover_videos
from aerojepa.train import build_video_dataloaders


def _write_video(path, num_frames: int, size: int = 48) -> None:
    """Write a deterministic solid-color clip; each frame's shade encodes its index."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (size, size))
    assert writer.isOpened(), "OpenCV could not open an mp4 writer in this environment"
    for i in range(num_frames):
        shade = int(255 * i / max(1, num_frames - 1))
        writer.write(np.full((size, size, 3), shade, dtype=np.uint8))
    writer.release()


def _write_telemetry(path, num_rows: int) -> None:
    header = ",".join(ACTION_COLUMNS)
    rows = np.arange(num_rows * len(ACTION_COLUMNS), dtype=np.float32)
    rows = rows.reshape(num_rows, len(ACTION_COLUMNS)) * 0.01
    np.savetxt(path, rows, delimiter=",", header=header, comments="")


@pytest.fixture()
def flights_dir(tmp_path):
    _write_video(tmp_path / "flight_001.mp4", num_frames=20)
    _write_telemetry(tmp_path / "flight_001.csv", num_rows=20)
    _write_video(tmp_path / "flight_002.mp4", num_frames=16)  # no telemetry
    return tmp_path


def test_discover_videos(flights_dir) -> None:
    paths = discover_videos(flights_dir)
    assert [p.name for p in paths] == ["flight_001.mp4", "flight_002.mp4"]


def test_uniform_mode_shapes(flights_dir) -> None:
    ds = VideoClipDataset(flights_dir, num_frames=8, img_size=32, mode="uniform")
    assert len(ds) == 2  # one clip per video
    frames, actions = ds[0]
    assert frames.shape == (8, 3, 32, 32)
    assert actions.shape == (8, len(ACTION_COLUMNS))
    assert frames.min() >= 0.0 and frames.max() <= 1.0


def test_sliding_windows_multiply_samples(flights_dir) -> None:
    ds = VideoClipDataset(
        flights_dir, num_frames=8, img_size=32, mode="sliding", stride=4
    )
    # flight_001 (20 frames): starts 0,4,8,12 -> 4 windows
    # flight_002 (16 frames): starts 0,4,8    -> 3 windows
    assert len(ds) == 7
    frames, actions = ds[0]
    assert frames.shape == (8, 3, 32, 32)
    assert actions.shape == (8, len(ACTION_COLUMNS))


def test_sliding_padding_for_short_clip(tmp_path) -> None:
    _write_video(tmp_path / "short.mp4", num_frames=5)
    ds = VideoClipDataset(
        tmp_path, num_frames=8, img_size=32, mode="sliding", stride=4, pad_short=True
    )
    assert len(ds) == 1
    frames, _ = ds[0]
    assert frames.shape == (8, 3, 32, 32)  # padded up to num_frames

    with pytest.raises(RuntimeError):  # nothing long enough, padding disabled
        VideoClipDataset(
            tmp_path, num_frames=8, img_size=32, mode="sliding", pad_short=False
        )


def test_missing_telemetry_is_zeros(flights_dir) -> None:
    ds = VideoClipDataset(flights_dir, num_frames=8, img_size=32, mode="uniform")
    # flight_002 has no CSV -> zero actions
    _, actions = ds[1]
    assert torch.count_nonzero(actions) == 0


def test_telemetry_alignment(flights_dir) -> None:
    table = load_telemetry_all(flights_dir / "flight_001.csv")
    assert table is not None and table.shape == (20, len(ACTION_COLUMNS))
    # A sliding window starting at frame 3 should pull rows 3..10.
    gathered = gather_telemetry(flights_dir / "flight_001.csv", list(range(3, 11)), 8)
    assert torch.allclose(gathered, table[3:11])


def test_explicit_video_paths_override(flights_dir) -> None:
    only = [flights_dir / "flight_002.mp4"]
    ds = VideoClipDataset(flights_dir, num_frames=4, img_size=32, video_paths=only)
    assert len(ds) == 1


def test_build_video_dataloaders_splits_by_video(flights_dir) -> None:
    data_cfg = {
        "data_dir": str(flights_dir),
        "num_frames": 8,
        "img_size": 32,
        "batch_size": 2,
        "window_mode": "sliding",
        "window_stride": 4,
        "val_fraction": 0.5,
        "num_workers": 0,
    }
    train_loader, val_loader = build_video_dataloaders(data_cfg)
    train_frames, train_actions = next(iter(train_loader))
    val_frames, _ = next(iter(val_loader))
    assert train_frames.shape[1:] == (8, 3, 32, 32)
    assert val_frames.shape[1:] == (8, 3, 32, 32)
    assert len(train_loader.dataset) > 0 and len(val_loader.dataset) > 0
