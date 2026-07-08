"""Transfer-curve experiment helpers — subset dirs, manifest, metrics extraction."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from aerojepa.data.video_dataset import discover_videos


def rollout_cosine_at_horizon(report: dict, horizon: int = 4) -> float | None:
    roll = report.get("real", report).get("rollout", report.get("rollout", {}))
    horizons = roll.get("horizon", [])
    cosines = roll.get("cosine", [])
    if horizon in horizons:
        return float(cosines[horizons.index(horizon)])
    return float(cosines[-1]) if cosines else None


def summarize_eval_point(
    train_clips: int,
    label: str,
    checkpoint: str,
    eval_report: dict,
) -> dict[str, Any]:
    real = eval_report["real"]["latent_prediction"]
    synth = eval_report["synthetic"]["latent_prediction"]
    return {
        "train_clips": train_clips,
        "label": label,
        "checkpoint": checkpoint,
        "real_latent_cosine": float(real["cosine"]),
        "synthetic_latent_cosine": float(synth["cosine"]),
        "sim_to_real_gap": float(eval_report["gap"]["latent_cosine"]),
        "real_smooth_l1": float(real["smooth_l1"]),
        "rollout_cosine_h4": rollout_cosine_at_horizon(eval_report, 4),
        "eval_data_dir": eval_report["data_dir"],
    }


def build_manifest(
    source_dir: str | Path,
    *,
    holdout_count: int = 3,
    train_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """Build a reproducible train/eval split and subset definitions."""
    source_dir = Path(source_dir)
    all_paths = discover_videos(source_dir)
    if len(all_paths) < 4:
        raise ValueError(
            f"Need at least 4 clips in {source_dir} for transfer curve; found {len(all_paths)}."
        )

    holdout_count = min(holdout_count, max(1, len(all_paths) // 5))
    holdout = all_paths[-holdout_count:]
    train_pool = all_paths[:-holdout_count]

    sizes = train_sizes or [1, 5, len(train_pool)]
    resolved_sizes: list[int] = []
    for n in sizes:
        capped = min(n, len(train_pool))
        if capped not in resolved_sizes:
            resolved_sizes.append(capped)
    resolved_sizes.sort()

    subsets: dict[str, list[str]] = {}
    for n in resolved_sizes:
        subset = train_pool[:n]
        subsets[str(n)] = [p.name for p in subset]

    return {
        "source_dir": str(source_dir.resolve()),
        "total_clips": len(all_paths),
        "holdout_count": holdout_count,
        "eval_holdout": [p.name for p in holdout],
        "train_pool_size": len(train_pool),
        "train_sizes": resolved_sizes,
        "subsets": subsets,
    }


def _link_clip(src_dir: Path, clip_name: str, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    src_video = src_dir / clip_name
    dest_video = dest_dir / clip_name
    if dest_video.exists() or dest_video.is_symlink():
        dest_video.unlink()
    dest_video.symlink_to(src_video.resolve())
    src_csv = src_video.with_suffix(".csv")
    if src_csv.is_file():
        dest_csv = dest_dir / src_csv.name
        if dest_csv.exists() or dest_csv.is_symlink():
            dest_csv.unlink()
        dest_csv.symlink_to(src_csv.resolve())


def prepare_subset_dirs(
    manifest: dict[str, Any],
    dest_root: str | Path,
    *,
    clean: bool = False,
) -> dict[str, Path]:
    """Create symlinked train subset dirs + fixed eval holdout under dest_root."""
    dest_root = Path(dest_root)
    source_dir = Path(manifest["source_dir"])

    if clean and dest_root.exists():
        shutil.rmtree(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)

    eval_dir = dest_root / "eval_holdout"
    eval_dir.mkdir(exist_ok=True)
    for name in manifest["eval_holdout"]:
        _link_clip(source_dir, name, eval_dir)

    dirs: dict[str, Path] = {"eval_holdout": eval_dir}
    for n_str, names in manifest["subsets"].items():
        train_dir = dest_root / f"train_n{n_str}"
        train_dir.mkdir(exist_ok=True)
        for name in names:
            _link_clip(source_dir, name, train_dir)
        dirs[f"train_n{n_str}"] = train_dir
    return dirs


def save_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return path
