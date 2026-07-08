"""Tests for transfer-curve manifest and metrics helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from aerojepa.eval.transfer_curve import (
    build_manifest,
    prepare_subset_dirs,
    rollout_cosine_at_horizon,
    summarize_eval_point,
)


def test_rollout_cosine_at_horizon():
    report = {"rollout": {"horizon": [1, 2, 3, 4], "cosine": [0.9, 0.91, 0.92, 0.93]}}
    assert rollout_cosine_at_horizon(report, 4) == pytest.approx(0.93)


def test_build_manifest_sizes(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(6):
        (src / f"clip_{i:02d}.mp4").write_bytes(b"x")
        (src / f"clip_{i:02d}.csv").write_text("dx,dy,d_altitude,d_yaw,d_pitch,d_roll\n0,0,0,0,0,0\n")

    manifest = build_manifest(src, holdout_count=2, train_sizes=[1, 3, 5])
    assert manifest["train_pool_size"] == 5
    assert manifest["train_sizes"] == [1, 3, 5]
    assert manifest["holdout_count"] == 1
    assert len(manifest["subsets"]["1"]) == 1


def test_prepare_subset_dirs(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(5):
        (src / f"c{i}.mp4").write_bytes(b"v")
    manifest = build_manifest(src, holdout_count=1, train_sizes=[1, 2])
    root = tmp_path / "curve"
    dirs = prepare_subset_dirs(manifest, root)
    assert (dirs["eval_holdout"] / manifest["eval_holdout"][0]).is_symlink()
    assert len(list(dirs["train_n1"].glob("*.mp4"))) == 1
    assert len(list(dirs["train_n2"].glob("*.mp4"))) == 2


def test_summarize_eval_point():
    report = {
        "data_dir": "data/x",
        "synthetic": {"latent_prediction": {"cosine": 0.99, "smooth_l1": 0.01}},
        "real": {"latent_prediction": {"cosine": 0.95, "smooth_l1": 0.03}},
        "gap": {"latent_cosine": 0.04},
        "rollout": {"horizon": [4], "cosine": [0.92]},
    }
    pt = summarize_eval_point(5, "5 clips", "ckpt.pt", report)
    assert pt["sim_to_real_gap"] == pytest.approx(0.04)
    assert pt["rollout_cosine_h4"] == pytest.approx(0.92)
