from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunLogger:
    """Minimal, dependency-free run logger.

    Every training run appends newline-delimited JSON to ``metrics.jsonl`` in
    its run directory. This keeps logs trivially greppable and plottable without
    a tracking server; Weights & Biases is optional and enabled per-config.
    """

    def __init__(self, run_dir: str | Path, use_wandb: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.use_wandb = use_wandb
        self._wandb = None

        if use_wandb:
            try:
                import wandb

                self._wandb = wandb
            except ImportError:
                self.use_wandb = False

    def init(self, config: dict[str, Any], project: str = "aerojepa") -> None:
        if self.use_wandb and self._wandb is not None:
            self._wandb.init(project=project, config=config, dir=str(self.run_dir))

    def log(self, metrics: dict[str, Any], step: int) -> None:
        record = {"step": step, **metrics}
        with self.metrics_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

        if self.use_wandb and self._wandb is not None:
            self._wandb.log(metrics, step=step)
