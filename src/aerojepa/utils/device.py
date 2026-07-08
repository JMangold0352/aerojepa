from __future__ import annotations

import os

import torch


def get_device(prefer: str = "auto") -> torch.device:
    """Pick a compute device, defaulting to the best one available.

    AeroJEPA is developed on Apple Silicon, so the Metal (MPS) backend is a
    first-class target. We set ``PYTORCH_ENABLE_MPS_FALLBACK`` so that the few
    ops MPS does not yet implement fall back to CPU instead of crashing.
    """
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if prefer == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device("cpu")
