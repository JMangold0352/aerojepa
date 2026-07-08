from __future__ import annotations

import json
from pathlib import Path

import pytest

# compare_ablations is a script; import its helpers directly.
import sys

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from visualizations.compare_ablations import _load_summary, generate_all


LEGACY_SUMMARY = {
    "baseline": {"objective": "masked", "cosine": 0.95, "smooth_l1": 0.04},
    "world_model": {"objective": "future", "cosine": 0.97, "smooth_l1": 0.03},
}

NEW_SUMMARY = {
    "meta": {"mode": "quick", "epochs": 20},
    "variants": {
        "baseline": {
            "objective": "masked",
            "latent_prediction": {"cosine": 0.95, "smooth_l1": 0.04},
            "rollout": {"horizon": [1, 2], "cosine": [0.9, 0.88]},
            "loop_analysis": {"per_loop_cosine": [0.85, 0.92]},
        },
        "loops_2": {
            "objective": "masked",
            "latent_prediction": {"cosine": 0.96, "smooth_l1": 0.03},
            "rollout": {"horizon": [1, 2], "cosine": [0.91, 0.89]},
            "loop_analysis": {"per_loop_cosine": [0.86, 0.94]},
        },
    },
}


def test_load_summary_legacy_format(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(LEGACY_SUMMARY))
    meta, variants = _load_summary(path)
    assert meta["mode"] == "legacy"
    assert variants["baseline"]["latent_prediction"]["cosine"] == 0.95


def test_load_summary_new_format(tmp_path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(NEW_SUMMARY))
    meta, variants = _load_summary(path)
    assert meta["epochs"] == 20
    assert "rollout" in variants["baseline"]


def test_generate_ablation_figures(tmp_path) -> None:
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(NEW_SUMMARY))
    out = tmp_path / "figures"
    paths = generate_all(summary, out, skip_gif=False)
    assert len(paths) >= 4
    assert (out / "01_latent_cosine_bar.png").exists()
    assert (out / "03_per_loop_cosine.png").exists()
    assert (out / "04_rollout_comparison.png").exists()
    assert (out / "README.md").exists()
