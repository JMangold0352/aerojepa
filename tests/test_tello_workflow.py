"""Tests for Tello preflight and transfer report helpers."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aerojepa.data.tello import ping_tello_host, preflight_check


def test_ping_tello_host_success():
    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        assert ping_tello_host("192.168.10.1") is True


def test_ping_tello_host_failure():
    with patch("subprocess.run") as run:
        run.return_value.returncode = 1
        assert ping_tello_host("192.168.10.1") is False


def test_preflight_fails_without_deps():
    with patch("aerojepa.data.tello.ping_tello_host", return_value=True):
        info = preflight_check(min_battery=20)
    assert info["ok"] is False
    assert info["opencv"] is True  # cv2 usually installed in dev env


def test_preflight_fails_when_wifi_unreachable():
    with patch("aerojepa.data.tello.ping_tello_host", return_value=False):
        info = preflight_check(min_battery=20)
    assert info["wifi_reachable"] is False
    assert info["ok"] is False


def test_compare_report_markdown():
    import sys
    from pathlib import Path

    scripts = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(scripts))
    import tello_compare_report as mod  # noqa: E402

    payload = {
        "generated_at": "2026-07-04",
        "baseline_checkpoint": "checkpoints/real_finetune_fast/latest.pt",
        "tello_checkpoint": "checkpoints/real_finetune_tello/latest.pt",
        "tello_data_dir": "data/flights_tello_128",
        "tello_data_eval": {
            "baseline_wilds_only": {
                "real_latent_cosine": 0.85,
                "rollout_cosine_at_4": 0.80,
                "sim_to_real_gap": 0.12,
            },
            "tello_finetuned": {
                "real_latent_cosine": 0.92,
                "rollout_cosine_at_4": 0.88,
                "sim_to_real_gap": 0.05,
            },
        },
    }
    md = mod.build_markdown(payload)
    assert "Wilds-only" in md
    assert "0.9200" in md or "0.92" in md
    assert "Tello transfer report" in md
