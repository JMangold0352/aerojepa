from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins on leaves)."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config, supporting single-level ``_base_`` inheritance.

    A config may set ``_base_: <relative-or-absolute path>`` to inherit from
    another config; the current file is deep-merged on top of the base. This is
    how we keep the many training recipes DRY -- for example, the looped and
    world-model variants inherit the whole synthetic-data recipe and only
    override the handful of predictor keys that actually differ.
    """
    path = Path(path)
    with open(path) as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    base_ref = cfg.pop("_base_", None)
    if base_ref is not None:
        base_path = Path(base_ref)
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        cfg = _deep_merge(load_config(base_path), cfg)
    return cfg
