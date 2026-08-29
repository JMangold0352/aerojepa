"""Tests for Tello shadow observer (no hardware, no flight commands)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from aerojepa.sim.tello_shadow import TelloShadowVehicle


def test_shadow_vehicle_step_refused() -> None:
    v = TelloShadowVehicle(img_size=32)
    with pytest.raises(RuntimeError, match="observer-only"):
        v.step(np.zeros(4, dtype=np.float32))


def test_new_shadow_files_grep_clean() -> None:
    """New shadow sources must not mention forbidden command APIs."""
    # Build tokens without embedding them as literals in this file.
    banned = ("take" + "off", "send_" + "rc", "l" + "and")
    roots = [
        Path("src/aerojepa/sim/tello_shadow.py"),
        Path("scripts/run_tello_shadow.py"),
    ]
    for path in roots:
        text = path.read_text().lower()
        for token in banned:
            assert token not in text, f"{path} contains banned token {token!r}"


def test_shadow_jsonl_schema_roundtrip(tmp_path: Path) -> None:
    row = {
        "t": 0.5,
        "vp": 0.0,
        "vq": 0.1,
        "vr": 0.0,
        "T": 0.39,
        "loop_ms": 40.0,
        "budget_ms": 25.0,
    }
    path = tmp_path / "demo_shadow.jsonl"
    path.write_text(json.dumps(row) + "\n")
    loaded = json.loads(path.read_text().strip())
    assert loaded["t"] == 0.5
    assert loaded["T"] == 0.39
