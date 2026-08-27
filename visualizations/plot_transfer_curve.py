#!/usr/bin/env python
"""Publication figure for the sim-to-real transfer curve experiment.

Reads ``results/transfer_curve/summary.json`` and writes:

  transfer_curve.png / transfer_curve.pdf - dual panel (gap + rollout @ h=4)

Examples::

    python visualizations/plot_transfer_curve.py
    python visualizations/plot_transfer_curve.py --summary results/transfer_curve/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from aerojepa.viz.style import DPI, PALETTE, apply_style


def plot_transfer_curve(summary_path: Path, out_dir: Path) -> tuple[Path, Path]:
    raw = json.loads(summary_path.read_text())
    points = sorted(raw["points"], key=lambda p: p["train_clips"])
    meta = raw.get("meta", {})

    xs = [p["train_clips"] for p in points]
    gaps = [p["sim_to_real_gap"] for p in points]
    rollouts = [p.get("rollout_cosine_h4") or float("nan") for p in points]
    real_cos = [p["real_latent_cosine"] for p in points]

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Panel A: sim-to-real gap (lower is better)
    ax = axes[0]
    ax.plot(xs, gaps, "o-", color=PALETTE["accent"], linewidth=2.2, markersize=9, label="Gap")
    ax.set_xlabel("Real clips used for fine-tuning")
    ax.set_ylabel("Sim-to-real latent cosine gap ↓")
    ax.set_title("Transfer gap vs data volume")
    ax.axhline(gaps[0], color=PALETTE["muted"], linestyle="--", linewidth=1, alpha=0.7,
               label="Synthetic-only baseline")
    if len(set(xs)) > 1:
        ax.set_xticks(xs)
    ax.legend(loc="upper right")

    holdout = meta.get("holdout_count", "?")
    ax.text(
        0.02, 0.02,
        f"{holdout} clips held out for eval",
        transform=ax.transAxes, fontsize=9, color=PALETTE["muted"],
    )

    # Panel B: rollout @ h=4 on held-out real clips (higher is better)
    ax = axes[1]
    ax.plot(xs, rollouts, "s-", color=PALETTE["target"], linewidth=2.2, markersize=9,
            label="Rollout @ h=4")
    ax.plot(xs, real_cos, "^--", color=PALETTE["looped"], linewidth=1.8, markersize=8,
            alpha=0.85, label="Real latent cosine")
    ax.set_xlabel("Real clips used for fine-tuning")
    ax.set_ylabel("Quality on held-out real clips ↑")
    ax.set_title("Rollout & latent quality vs data volume")
    if len(set(xs)) > 1:
        ax.set_xticks(xs)
    ax.legend(loc="lower right")

    note = meta.get("note", "")
    if note:
        fig.suptitle(
            "Sim-to-real transfer curve (Wilds Parrot footage)",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.text(0.5, -0.02, note, ha="center", fontsize=9, color=PALETTE["muted"])

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "transfer_curve.png"
    pdf = out_dir / "transfer_curve.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/transfer_curve/summary.json")
    parser.add_argument("--out-dir", default="results/transfer_curve")
    args = parser.parse_args()
    png, pdf = plot_transfer_curve(Path(args.summary), Path(args.out_dir))
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
