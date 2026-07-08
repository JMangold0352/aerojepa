from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from aerojepa.data.telemetry import ACTION_COLUMNS, gather_telemetry

# Real-data hook. This is the drop-in replacement for the synthetic dataset once
# you have footage (self-collected Tello clips, MotionScape, AeroVerse, etc.).
# It intentionally mirrors the synthetic dataset's ``(frames, actions)`` output
# so training and evaluation code never has to know which source it is using.
#
# Two sampling modes cover the two things real footage throws at you:
#   * "uniform"  -- one clip per video, ``num_frames`` frames spread across the
#                   whole file. Good for a folder of many short, similar clips.
#   * "sliding"  -- many overlapping fixed-length windows per video. Good for a
#                   few long flights, and it multiplies a small corpus into many
#                   training samples. Clips shorter than ``num_frames`` are
#                   padded by repeating the last frame (``pad_short=True``).
#
# OpenCV is an optional dependency (``pip install opencv-python``); we import it
# lazily so the core install stays lean.

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")


def discover_videos(data_dir: str | Path) -> list[Path]:
    """Return the sorted list of video files in ``data_dir``.

    Sorting keeps train/val splits deterministic across runs and machines.
    """
    data_dir = Path(data_dir)
    return sorted(p for p in data_dir.glob("*") if p.suffix.lower() in VIDEO_SUFFIXES)


def _require_cv2():
    try:
        import cv2  # noqa: PLC0415 - optional dependency, imported lazily
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "VideoClipDataset needs opencv-python. Install with `pip install opencv-python`."
        ) from exc
    return cv2


def _frame_count(path: Path) -> int:
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return max(0, total)


class VideoClipDataset(Dataset):
    """Load fixed-length clips from a folder (or explicit list) of video files.

    Directory convention::

        data_dir/
        ├── flight_001.mp4
        ├── flight_001.csv   # optional telemetry, columns = ACTION_COLUMNS
        ├── flight_002.mp4
        └── ...

    Every item is a ``(frames, actions)`` pair with ``frames`` shaped
    ``(num_frames, C, img_size, img_size)`` in ``[0, 1]`` and ``actions`` shaped
    ``(num_frames, 6)``. Telemetry is read from the sibling ``.csv`` when present
    and kept index-aligned with the sampled frames; otherwise it is zeros.

    Args:
        data_dir: Folder to scan for videos (used when ``video_paths`` is None).
        num_frames: Clip length handed to the model.
        img_size: Square resolution every frame is resized to.
        mode: ``"uniform"`` (one clip per video) or ``"sliding"`` (many windows).
        stride: Step between sliding windows (defaults to ``num_frames``, i.e.
            non-overlapping). Ignored in uniform mode.
        pad_short: If True, videos shorter than ``num_frames`` are padded by
            repeating the last frame; if False they are skipped (sliding mode).
        video_paths: Optional explicit list of files, used instead of scanning
            ``data_dir`` -- this is how the trainer holds out whole videos for
            validation without leaking windows across the split.
    """

    VIDEO_SUFFIXES = VIDEO_SUFFIXES

    def __init__(
        self,
        data_dir: str | Path,
        num_frames: int = 8,
        img_size: int = 64,
        mode: str = "uniform",
        stride: int | None = None,
        pad_short: bool = True,
        video_paths: list[Path] | None = None,
    ) -> None:
        if mode not in ("uniform", "sliding"):
            raise ValueError(f"mode must be 'uniform' or 'sliding', got {mode!r}")
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.img_size = img_size
        self.mode = mode
        self.stride = stride if stride and stride > 0 else num_frames
        self.pad_short = pad_short

        self.paths = list(video_paths) if video_paths is not None else discover_videos(self.data_dir)
        if not self.paths:
            raise FileNotFoundError(
                f"No videos ({', '.join(VIDEO_SUFFIXES)}) found in {self.data_dir}. "
                "Point data.data_dir at a folder of clips, or use the synthetic generator."
            )
        # Each sample is (video_path, start_frame). start_frame is None in
        # uniform mode (frames are spread across the whole video instead).
        self._samples = self._build_index()
        if not self._samples:
            raise RuntimeError(
                f"No clips could be formed from {len(self.paths)} video(s) with "
                f"num_frames={num_frames}, mode={mode}. Try mode='uniform' or "
                "pad_short=True for very short footage."
            )

    def _build_index(self) -> list[tuple[Path, int | None]]:
        if self.mode == "uniform":
            return [(p, None) for p in self.paths]

        samples: list[tuple[Path, int | None]] = []
        for path in self.paths:
            total = _frame_count(path)
            if total <= 0:
                continue
            if total < self.num_frames:
                if self.pad_short:
                    samples.append((path, 0))
                continue
            last_start = total - self.num_frames
            starts = list(range(0, last_start + 1, self.stride))
            if starts and starts[-1] != last_start:
                starts.append(last_start)  # always cover the tail of the clip
            samples.extend((path, s) for s in starts)
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def _frame_indices(self, path: Path, start: int | None) -> list[int]:
        if start is None:  # uniform: spread num_frames across the whole video
            total = _frame_count(path) or self.num_frames
            return (
                torch.linspace(0, max(0, total - 1), self.num_frames)
                .round()
                .long()
                .tolist()
            )
        return list(range(start, start + self.num_frames))  # sliding: consecutive

    def _read_indices(self, path: Path, indices: list[int]) -> torch.Tensor:
        cv2 = _require_cv2()
        cap = cv2.VideoCapture(str(path))
        frames: list[torch.Tensor] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                if frames:  # ran past the end: repeat the last valid frame
                    frames.append(frames[-1].clone())
                    continue
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame).float().permute(2, 0, 1) / 255.0)
        cap.release()

        if not frames:
            raise RuntimeError(f"Could not read any frames from {path}.")
        while len(frames) < len(indices):  # pad short clips to full length
            frames.append(frames[-1].clone())

        clip = torch.stack(frames)  # (T, C, H, W)
        return F.interpolate(
            clip, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, start = self._samples[idx]
        indices = self._frame_indices(path, start)
        frames = self._read_indices(path, indices)
        actions = gather_telemetry(path.with_suffix(".csv"), indices, self.num_frames)
        assert actions.shape == (self.num_frames, len(ACTION_COLUMNS))
        return frames, actions
