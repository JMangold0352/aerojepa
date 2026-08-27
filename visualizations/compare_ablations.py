#!/usr/bin/env python
"""Publication figures from ablation results (looped-jepa style).

Reads ``results/ablations/summary.json`` (or per-variant JSON files) and writes
high-DPI figures to ``visualizations/figures/ablations/``:

  01_latent_cosine_bar.png      - headline metric across variants
  02_smooth_l1_bar.png          - complementary distance metric
  03_per_loop_cosine.png        - refinement gain per loop (looped variants)
  04_rollout_comparison.png     - rollout cosine vs horizon, all variants
  05_rollout_panel.png          - side-by-side rollout curves (faceted)
  06_rollout_comparison.gif     - animated build-up of rollout curves

Examples::

    python visualizations/compare_ablations.py
    python visualizations/compare_ablations.py --summary results/ablations/summary.json
    python visualizations/compare_ablations.py --fast   # skip GIF
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

# Make src/ importable when run directly.
_ROOT = Path(__file__).resolve().parents[1]
for _p in (_ROOT / "src", _ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from aerojepa.viz.style import DPI, PALETTE, apply_style

# Display order and colors - baseline blue, looped orange, world model green.
VARIANT_ORDER = ["baseline", "loops_2", "loops_3", "world_model"]
VARIANT_LABELS = {
    "baseline": "baseline\n(feed-forward)",
    "loops_2": "looped\n(2 loops)",
    "loops_3": "looped\n(3 loops)",
    "world_model": "world model\n(future obj.)",
}
VARIANT_COLORS = {
    "baseline": PALETTE["baseline"],
    "loops_2": PALETTE["looped"],
    "loops_3": "#E17C47",
    "world_model": PALETTE["target"],
}


def _load_summary(path: Path) -> tuple[dict, dict]:
    """Return ``(meta, variants)`` from summary JSON (new or legacy format)."""
    raw = json.loads(path.read_text())
    if "variants" in raw:
        return raw.get("meta", {}), raw["variants"]
    # Legacy flat format: { "baseline": { "cosine": ..., "smooth_l1": ... }, ... }
    variants = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict) or "cosine" not in entry:
            continue
        variants[name] = {
            "objective": entry.get("objective", "masked"),
            "latent_prediction": {
                "cosine": entry["cosine"],
                "smooth_l1": entry.get("smooth_l1", 0.0),
            },
        }
    return {"mode": "legacy", "epochs": "?"}, variants


def _ordered_variants(variants: dict) -> list[str]:
    known = [v for v in VARIANT_ORDER if v in variants]
    extra = [v for v in variants if v not in VARIANT_ORDER]
    return known + extra


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_latent_bars(variants: dict, out_dir: Path, meta: dict) -> Path:
    apply_style()
    names = _ordered_variants(variants)
    cosines = [variants[n]["latent_prediction"]["cosine"] for n in names]
    labels = [VARIANT_LABELS.get(n, n) for n in names]
    colors = [VARIANT_COLORS.get(n, PALETTE["muted"]) for n in names]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(names))
    bars = ax.bar(x, cosines, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("latent cosine similarity")
    ax.set_ylim(max(0.0, min(cosines) - 0.02), min(1.0, max(cosines) + 0.01))
    epochs = meta.get("epochs", "?")
    mode = meta.get("mode", "")
    ax.set_title(f"Ablation: latent prediction quality ({mode}, {epochs} epochs)")
    for bar, val in zip(bars, cosines):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    return _save(fig, out_dir / "01_latent_cosine_bar.png")


def plot_smooth_l1_bars(variants: dict, out_dir: Path, meta: dict) -> Path:
    apply_style()
    names = _ordered_variants(variants)
    values = [variants[n]["latent_prediction"]["smooth_l1"] for n in names]
    labels = [VARIANT_LABELS.get(n, n) for n in names]
    colors = [VARIANT_COLORS.get(n, PALETTE["muted"]) for n in names]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    x = np.arange(len(names))
    bars = ax.bar(x, values, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("smooth-L1 distance (lower is better)")
    epochs = meta.get("epochs", "?")
    ax.set_title(f"Ablation: latent prediction error ({meta.get('mode', '')}, {epochs} ep)")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    return _save(fig, out_dir / "02_smooth_l1_bar.png")


def plot_per_loop_cosine(variants: dict, out_dir: Path, meta: dict) -> Path | None:
    looped = {
        n: v["loop_analysis"]["per_loop_cosine"]
        for n, v in variants.items()
        if v.get("loop_analysis", {}).get("per_loop_cosine")
    }
    if not looped:
        return None

    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for name in _ordered_variants(looped):
        curve = looped[name]
        loops = list(range(1, len(curve) + 1))
        ax.plot(
            loops,
            curve,
            "-o",
            color=VARIANT_COLORS.get(name, PALETTE["muted"]),
            linewidth=2.0,
            label=VARIANT_LABELS.get(name, name).replace("\n", " "),
        )
        # Annotate gain from loop 1 -> last
        gain = curve[-1] - curve[0]
        ax.annotate(
            f"+{gain:.3f}",
            xy=(loops[-1], curve[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=VARIANT_COLORS.get(name, PALETTE["muted"]),
        )

    ax.set_xlabel("refinement loop")
    ax.set_ylabel("latent cosine similarity")
    ax.set_title(f"Per-loop refinement gain ({meta.get('mode', '')}, {meta.get('epochs', '?')} ep)")
    ax.legend(loc="lower right")
    ymin = min(min(c) for c in looped.values()) - 0.02
    ax.set_ylim(max(0.0, ymin), 1.02)
    return _save(fig, out_dir / "03_per_loop_cosine.png")


def _rollout_data(variants: dict) -> dict[str, dict]:
    out = {}
    for name, v in variants.items():
        roll = v.get("rollout")
        if roll and roll.get("horizon") and roll.get("cosine"):
            out[name] = roll
    return out


def plot_rollout_comparison(rollouts: dict, out_dir: Path, meta: dict) -> Path | None:
    if not rollouts:
        return None
    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for name in _ordered_variants(rollouts):
        roll = rollouts[name]
        ax.plot(
            roll["horizon"],
            roll["cosine"],
            "-o",
            color=VARIANT_COLORS.get(name, PALETTE["muted"]),
            linewidth=2.0,
            label=VARIANT_LABELS.get(name, name).replace("\n", " "),
        )
    ax.set_xlabel("prediction horizon (frames ahead)")
    ax.set_ylabel("latent cosine similarity")
    ax.set_title(f"Rollout comparison ({meta.get('mode', '')}, {meta.get('epochs', '?')} ep)")
    ax.legend(loc="best", fontsize=9)
    ax.set_ylim(
        max(0.0, min(min(r["cosine"]) for r in rollouts.values()) - 0.03),
        1.02,
    )
    return _save(fig, out_dir / "04_rollout_comparison.png")


def plot_rollout_panel(rollouts: dict, out_dir: Path, meta: dict) -> Path | None:
    """Side-by-side small multiples - one rollout curve per variant."""
    if not rollouts:
        return None
    names = _ordered_variants(rollouts)
    n = len(names)
    apply_style()
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, name in zip(axes, names):
        roll = rollouts[name]
        color = VARIANT_COLORS.get(name, PALETTE["muted"])
        ax.plot(roll["horizon"], roll["cosine"], "-o", color=color, linewidth=2.0)
        ax.set_title(VARIANT_LABELS.get(name, name), fontsize=10)
        ax.set_xlabel("horizon")
        ax.set_ylim(
            max(0.0, min(min(r["cosine"]) for r in rollouts.values()) - 0.03),
            1.02,
        )
    axes[0].set_ylabel("latent cosine")
    fig.suptitle(
        f"Rollout by variant ({meta.get('mode', '')}, {meta.get('epochs', '?')} ep)",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    return _save(fig, out_dir / "05_rollout_panel.png")


def plot_rollout_gif(rollouts: dict, out_dir: Path, meta: dict) -> Path | None:
    """Animated GIF: rollout curves appear one variant at a time."""
    if not rollouts:
        return None
    try:
        from PIL import Image
    except ImportError:
        return None

    apply_style()
    names = _ordered_variants(rollouts)
    all_cos = [c for r in rollouts.values() for c in r["cosine"]]
    ylo = max(0.0, min(all_cos) - 0.03)
    frames: list[Image.Image] = []

    for step in range(1, len(names) + 1):
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        for name in names[:step]:
            roll = rollouts[name]
            ax.plot(
                roll["horizon"],
                roll["cosine"],
                "-o",
                color=VARIANT_COLORS.get(name, PALETTE["muted"]),
                linewidth=2.0,
                label=VARIANT_LABELS.get(name, name).replace("\n", " "),
            )
        ax.set_xlabel("prediction horizon (frames ahead)")
        ax.set_ylabel("latent cosine similarity")
        ax.set_title(f"Rollout comparison - {step}/{len(names)} variants")
        ax.set_ylim(ylo, 1.02)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()
        tmp = out_dir / "_gif_frame.png"
        fig.savefig(tmp, dpi=120, bbox_inches="tight")
        plt.close(fig)
        frames.append(Image.open(tmp).convert("P", palette=Image.ADAPTIVE))

    out = out_dir / "06_rollout_comparison.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=900,
        loop=0,
        disposal=2,
    )
    (out_dir / "_gif_frame.png").unlink(missing_ok=True)
    return out


def generate_all(summary_path: Path, out_dir: Path, skip_gif: bool = False) -> list[Path]:
    meta, variants = _load_summary(summary_path)
    if not variants:
        raise ValueError(f"No variants found in {summary_path}")

    written: list[Path] = []
    written.append(plot_latent_bars(variants, out_dir, meta))
    written.append(plot_smooth_l1_bars(variants, out_dir, meta))

    p = plot_per_loop_cosine(variants, out_dir, meta)
    if p:
        written.append(p)

    rollouts = _rollout_data(variants)
    p = plot_rollout_comparison(rollouts, out_dir, meta)
    if p:
        written.append(p)
    p = plot_rollout_panel(rollouts, out_dir, meta)
    if p:
        written.append(p)
    if not skip_gif:
        p = plot_rollout_gif(rollouts, out_dir, meta)
        if p:
            written.append(p)

    # Human-readable table alongside figures.
    md_lines = [
        "# Ablation summary",
        "",
        f"Source: `{summary_path}`",
        f"Mode: **{meta.get('mode', '?')}** · Epochs: **{meta.get('epochs', '?')}**",
        "",
        "| Variant | Objective | Latent cosine | Smooth-L1 | Rollout @ last h |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name in _ordered_variants(variants):
        v = variants[name]
        lp = v["latent_prediction"]
        roll = v.get("rollout", {})
        rh = roll["cosine"][-1] if roll.get("cosine") else float("nan")
        md_lines.append(
            f"| {name} | {v.get('objective', '?')} | {lp['cosine']:.4f} | "
            f"{lp['smooth_l1']:.4f} | {rh:.4f} |"
        )
    md_lines += ["", "## Figures", ""]
    for p in written:
        md_lines.append(f"- `{p.name}`")
    (out_dir / "README.md").write_text("\n".join(md_lines) + "\n")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="results/ablations/summary.json")
    parser.add_argument("--out-dir", default="visualizations/figures/ablations")
    parser.add_argument("--fast", action="store_true", help="Skip animated GIF.")
    args = parser.parse_args()

    paths = generate_all(Path(args.summary), Path(args.out_dir), skip_gif=args.fast)
    print(f"Wrote {len(paths)} figure(s) to {args.out_dir}:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
