"""Backend dispatch for video preprocess (OpenCV default, optional Rust).

Design: docs/NATIVE_PREPROCESS.md

This module is the single switch for all crowds:
- No Rust installed → OpenCV (existing path)
- Rust wheel installed → try native; on failure, fall back to OpenCV
- Explicit override via ``AEROJEPA_PREPROCESS_BACKEND`` = auto|opencv|rust
  or ``set_backend(...)`` / CLI ``--backend``.
"""

from __future__ import annotations

import os
from typing import Literal

BackendName = Literal["auto", "opencv", "rust"]

# Process-level override (CLI sets this). Env var used when this is None.
_backend_override: BackendName | None = None


def set_backend(name: BackendName | str) -> None:
    """Set process-level backend preference (``auto`` | ``opencv`` | ``rust``)."""
    global _backend_override
    name = str(name).strip().lower()
    if name not in ("auto", "opencv", "rust"):
        raise ValueError(f"unknown preprocess backend: {name!r}")
    _backend_override = name  # type: ignore[assignment]


def requested_backend() -> BackendName:
    if _backend_override is not None:
        return _backend_override
    raw = os.environ.get("AEROJEPA_PREPROCESS_BACKEND", "auto").strip().lower()
    if raw in ("auto", "opencv", "rust"):
        return raw  # type: ignore[return-value]
    return "auto"


def rust_available() -> bool:
    try:
        import aerojepa_preprocess  # noqa: F401
        return True
    except ImportError:
        return False


def active_backend() -> str:
    """Resolve which backend will be used for frame-index selection."""
    req = requested_backend()
    if req == "opencv":
        return "opencv"
    if req == "rust":
        if not rust_available():
            raise ImportError(
                "preprocess backend=rust but aerojepa_preprocess is not installed. "
                "Build with: cd native/aerojepa-preprocess && maturin develop --release"
            )
        return "rust"
    # auto
    return "rust" if rust_available() else "opencv"


def select_indices_native_or_python(
    src_frames: int,
    src_fps: float,
    target_fps: int,
    max_seconds: float | None,
) -> list[int]:
    """Phase A: prefer Rust ``select_indices`` when available (parity-tested).

    Falls back to the Python implementation in ``preprocess._select_indices``.
    """
    if active_backend() == "rust":
        try:
            import aerojepa_preprocess as native

            return list(
                native.select_indices(src_frames, float(src_fps), int(target_fps), max_seconds)
            )
        except Exception:
            pass
    from aerojepa.data.preprocess import _select_indices

    return _select_indices(src_frames, src_fps, target_fps, max_seconds)
