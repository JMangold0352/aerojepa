from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from aerojepa.viz.style import PALETTE, apply_style


def _save(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_clip_frames(frames: torch.Tensor, path: str | Path, title: str = "Drone clip") -> Path:
    """Contact sheet of a clip's frames over time."""
    apply_style()
    t = frames.shape[0]
    fig, axes = plt.subplots(1, t, figsize=(1.8 * t, 2.2))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        ax.imshow(frames[i].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        ax.set_title(f"t={i}", fontsize=9)
        ax.axis("off")
        ax.grid(False)
    fig.suptitle(title)
    return _save(fig, path)


def plot_rollout_curve(rollout: dict, path: str | Path) -> Path:
    """Prediction quality as a function of how far ahead we predict."""
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(rollout["horizon"], rollout["cosine"], "-o", color=PALETTE["looped"], label="cosine to teacher")
    ax.set_xlabel("prediction horizon (frames ahead)")
    ax.set_ylabel("latent cosine similarity")
    ax.set_title("World-model rollout: accuracy vs horizon")
    ax.set_ylim(min(0.0, min(rollout["cosine"]) - 0.05), 1.02)
    ax.legend()
    return _save(fig, path)


def plot_per_loop_cosine(loop: dict, path: str | Path) -> Path:
    """Does each extra refinement loop actually improve the prediction?"""
    apply_style()
    loops = list(range(1, len(loop["per_loop_cosine"]) + 1))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(loops, loop["per_loop_cosine"], "-o", color=PALETTE["looped"])
    ax.set_xlabel("refinement loop")
    ax.set_ylabel("latent cosine similarity")
    ax.set_title("Per-loop refinement of the predictor")
    ax.set_xticks(loops)
    return _save(fig, path)


def plot_exit_distribution(loop: dict, path: str | Path) -> Path:
    """How the learned exit gate allocates compute across depths."""
    apply_style()
    dist = loop.get("exit_distribution") or []
    fig, ax = plt.subplots(figsize=(5.5, 4))
    if dist:
        loops = list(range(1, len(dist) + 1))
        ax.bar(loops, dist, color=PALETTE["baseline"])
        ax.set_xticks(loops)
        ax.set_title(f"Adaptive exit depth (expected loops = {loop['expected_loops']:.2f})")
    else:
        ax.text(0.5, 0.5, "No exit gate in this model", ha="center", va="center")
        ax.set_title("Adaptive exit depth")
    ax.set_xlabel("exit loop")
    ax.set_ylabel("fraction of clips")
    return _save(fig, path)


def plot_latent_trajectory(per_frame_latents: torch.Tensor, path: str | Path) -> Path:
    """2D PCA of the per-frame latent, tracing how the scene evolves in time.

    ``per_frame_latents`` is ``(T, D)``. PCA is computed with a plain SVD so we
    avoid a scikit-learn dependency for a two-component projection.
    """
    apply_style()
    x = per_frame_latents.cpu().float()
    x = x - x.mean(dim=0, keepdim=True)
    _, _, v = torch.linalg.svd(x, full_matrices=False)
    coords = (x @ v[:2].T).numpy()

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(coords[:, 0], coords[:, 1], "-", color=PALETTE["muted"], zorder=1)
    sc = ax.scatter(coords[:, 0], coords[:, 1], c=range(len(coords)), cmap="viridis", zorder=2)
    for i, (px, py) in enumerate(coords):
        ax.annotate(str(i), (px, py), fontsize=8, xytext=(3, 3), textcoords="offset points")
    fig.colorbar(sc, ax=ax, label="frame index")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Latent trajectory over time")
    return _save(fig, path)


def plot_attention_matrix(
    attn: torch.Tensor, num_temporal: int, num_context: int, path: str | Path
) -> Path:
    """Predictor self-attention, with frame boundaries drawn in.

    ``attn`` is a ``(N, N)`` averaged attention map. Gridlines mark where one
    frame ends and the next begins, so it is easy to see whether the predictor
    reasons *across time* (off-diagonal frame blocks) or only within a frame.
    """
    apply_style()
    a = attn.cpu().numpy()
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(a, cmap="magma")
    fig.colorbar(im, ax=ax, label="attention weight")
    ax.axhline(num_context - 0.5, color="white", lw=1.0, alpha=0.7)
    ax.axvline(num_context - 0.5, color="white", lw=1.0, alpha=0.7)
    ax.set_title("Predictor attention (context | targets)")
    ax.set_xlabel("key token")
    ax.set_ylabel("query token")
    ax.grid(False)
    return _save(fig, path)


def plot_comparison_bars(scores: dict[str, float], path: str | Path, title: str = "Latent cosine by variant") -> Path:
    """Simple bar chart summarizing a metric across model variants."""
    apply_style()
    names = list(scores.keys())
    values = [scores[n] for n in names]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(names, values, color=PALETTE["baseline"])
    ax.set_ylabel("latent cosine similarity")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return _save(fig, path)
