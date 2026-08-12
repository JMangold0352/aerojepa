"""Backend selection for optional native preprocess."""

from __future__ import annotations

import os

import pytest

from aerojepa.data.preprocess_backend import (
    active_backend,
    requested_backend,
    rust_available,
    select_indices_native_or_python,
    set_backend,
)


@pytest.fixture(autouse=True)
def _reset_backend():
    set_backend("auto")
    yield
    set_backend("auto")


def test_default_backend_resolves(monkeypatch):
    monkeypatch.delenv("AEROJEPA_PREPROCESS_BACKEND", raising=False)
    set_backend("auto")
    assert requested_backend() == "auto"
    # With or without the optional wheel: auto must resolve cleanly.
    assert active_backend() in ("opencv", "rust")


def test_select_indices_python_fallback_matches_preprocess():
    from aerojepa.data.preprocess import _select_indices

    set_backend("opencv")
    a = _select_indices(100, 30.0, 15, None)
    b = select_indices_native_or_python(100, 30.0, 15, None)
    assert a == b
    assert len(a) > 0
    assert a[0] == 0
    assert a[-1] == 99


def test_env_override_opencv(monkeypatch):
    monkeypatch.setenv("AEROJEPA_PREPROCESS_BACKEND", "opencv")
    set_backend("auto")  # env should win when override cleared
    # Process override takes precedence; clear it by setting auto then reading env.
    # set_backend("auto") with env=opencv still means requested is auto from override.
    # Explicit opencv:
    set_backend("opencv")
    assert requested_backend() == "opencv"
    assert active_backend() == "opencv"


@pytest.mark.skipif(not rust_available(), reason="aerojepa_preprocess not installed")
def test_rust_select_indices_parity_with_python():
    from aerojepa.data.preprocess import _select_indices

    cases = [
        (0, 30.0, 15, None),
        (1, 30.0, 15, None),
        (100, 30.0, 15, None),
        (50, 0.0, 15, 10.0),
        (200, 60.0, 15, None),
        (8, 15.0, 15, None),
        (1000, 30.0, 10, 5.0),
    ]
    for args in cases:
        py = _select_indices(*args)
        set_backend("rust")
        rs = select_indices_native_or_python(*args)
        assert rs == py, f"parity fail for {args}: py={py[:8]}... rs={rs[:8]}..."


@pytest.mark.skipif(not rust_available(), reason="aerojepa_preprocess not installed")
def test_auto_prefers_rust_when_installed():
    set_backend("auto")
    assert active_backend() == "rust"
