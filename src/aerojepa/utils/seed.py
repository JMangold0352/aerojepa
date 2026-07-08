from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch so runs (and synthetic data) reproduce."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
