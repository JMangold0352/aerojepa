"""Compare regular vs looped predictor ablations -- the headline research question.

Loads both ablation result dirs and produces:
- A combined metrics table (naive/plain/structured x regular/looped).
- A bar chart comparing structured-prober accuracy under regular vs looped.
- A JSON summary answering: does the looped predictor's refined latent rollout
  improve metric-trajectory accuracy?

Usage:
    python research/prober/scripts/compare_regular_looped.py \
        --regular research/prober/results/prober_regular_ablation \
        --looped research/prober/results/prober_looped_ablation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regular", default="research/prober/results/prober_regular_ablation")
    parser.add_argument("--looped", default="research/prober/results/prober_looped_ablation")
    parser.add_argument("--output-dir", default="research/prober/results/regular_vs_looped")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    reg = load_summary(Path(args.regular) / "summary.json")
    loo = load_summary(Path(args.looped) / "summary.json")

    arms = ["naive", "plain", "structured"]
    metrics = ["position_rmse", "attitude_rmse_deg", "velocity_rmse"]

    # Combined table.
    table = {}
    for arm in arms:
        table[arm] = {
            "regular": {m: (reg[arm][f"{m}_mean"], reg[arm][f"{m}_std"]) for m in metrics},
            "looped": {m: (loo[arm][f"{m}_mean"], loo[arm][f"{m}_std"]) for m in metrics},
        }

    # Headline question: does looped beat regular for the structured prober?
    s_reg_pos = reg["structured"]["position_rmse_mean"]
    s_loo_pos = loo["structured"]["position_rmse_mean"]
    s_reg_att = reg["structured"]["attitude_rmse_deg_mean"]
    s_loo_att = loo["structured"]["attitude_rmse_deg_mean"]
    headline = {
        "question": "Does the looped predictor's refined latent rollout improve metric-trajectory accuracy (structured prober)?",
        "structured_position_rmse_regular": s_reg_pos,
        "structured_position_rmse_looped": s_loo_pos,
        "structured_position_looped_better": bool(s_loo_pos < s_reg_pos),
        "structured_attitude_rmse_deg_regular": s_reg_att,
        "structured_attitude_rmse_deg_looped": s_loo_att,
        "structured_attitude_looped_better": bool(s_loo_att < s_reg_att),
    }

    with open(out_dir / "comparison.json", "w") as f:
        json.dump({"table": table, "headline": headline}, f, indent=2)

    # Print the combined table.
    print("[compare] === REGULAR vs LOOPED (mean +/- std) ===")
    print(f"{'arm':<12} {'metric':<20} {'regular':<22} {'looped':<22} {'looped better?'}")
    for arm in arms:
        for m in metrics:
            r_mean, r_std = table[arm]["regular"][m]
            l_mean, l_std = table[arm]["looped"][m]
            better = "yes" if l_mean < r_mean else "no"
            print(
                f"{arm:<12} {m:<20} "
                f"{r_mean:.4f} +/- {r_std:.4f}    "
                f"{l_mean:.4f} +/- {l_std:.4f}    {better}"
            )
    print(f"\n[compare] headline: looped better on position? {headline['structured_position_looped_better']} "
          f"({s_reg_pos:.4f} -> {s_loo_pos:.4f})")
    print(f"[compare] headline: looped better on attitude? {headline['structured_attitude_looped_better']} "
          f"({s_reg_att:.4f} -> {s_loo_att:.4f})")

    # Bar chart: structured prober, regular vs looped, all 3 metrics.
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    for i, m in enumerate(metrics):
        means_reg = [table[a]["regular"][m][0] for a in arms]
        stds_reg = [table[a]["regular"][m][1] for a in arms]
        means_loo = [table[a]["looped"][m][0] for a in arms]
        stds_loo = [table[a]["looped"][m][1] for a in arms]
        x = np.arange(len(arms))
        w = 0.35
        ax[i].bar(x - w/2, means_reg, w, yerr=stds_reg, label="regular", color="#1f77b4", alpha=0.8)
        ax[i].bar(x + w/2, means_loo, w, yerr=stds_loo, label="looped", color="#ff7f0e", alpha=0.8)
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(arms)
        ax[i].set_title(m)
        ax[i].legend()
        ax[i].grid(True, alpha=0.3, axis="y")
    fig.suptitle("Regular vs looped predictor: does adaptive compute help metric groundability?")
    fig.tight_layout()
    fig.savefig(fig_dir / "regular_vs_looped.png", dpi=120)
    print(f"[compare] saved figures to {fig_dir}")
    print(f"[compare] done. results in {out_dir}")


if __name__ == "__main__":
    main()
