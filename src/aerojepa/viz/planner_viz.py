from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Visualizing a plan: turn the "observed context -> imagined/executed rollout"
# story into a single animated GIF, plus a static figure of the candidate
# trajectories the planner compared. Uses only Pillow + matplotlib (already
# dependencies) so the demo needs no extra packages.


def _to_uint8(frame: torch.Tensor) -> np.ndarray:
    return (frame.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype("uint8")


def render_plan_gif(
    context_frames: torch.Tensor,
    planned_frames: torch.Tensor,
    out_path: str | Path,
    coherence: float | None = None,
    cost: float | None = None,
    upscale: int = 6,
    duration_ms: int = 250,
) -> Path:
    """Write an annotated GIF: context frames, then the executed plan.

    ``context_frames`` / ``planned_frames`` are ``(T, C, H, W)`` tensors in
    ``[0, 1]``. Each frame gets a colored banner (blue = observed context, orange
    = planned rollout) and the final frame shows the plan's coherence/cost.
    """
    from PIL import Image, ImageDraw

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ctx = [(_to_uint8(f), "context") for f in context_frames]
    plan = [(_to_uint8(f), "planned") for f in planned_frames]
    banner = {"context": (54, 100, 227), "planned": (230, 126, 34)}

    images: list["Image.Image"] = []
    total = len(plan)
    for i, (arr, phase) in enumerate(ctx + plan):
        img = Image.fromarray(arr).resize(
            (arr.shape[1] * upscale, arr.shape[0] * upscale), Image.NEAREST
        )
        img = img.convert("RGB")
        draw = ImageDraw.Draw(img)
        bar_h = max(14, img.height // 10)
        draw.rectangle([0, 0, img.width, bar_h], fill=banner[phase])
        label = "OBSERVED" if phase == "context" else f"PLANNED  {i - len(ctx) + 1}/{total}"
        draw.text((4, 2), label, fill=(255, 255, 255))
        if phase == "planned" and i == len(ctx) + total - 1:
            note = []
            if coherence is not None:
                note.append(f"coherence {coherence:.2f}")
            if cost is not None:
                note.append(f"cost {cost:.3f}")
            if note:
                draw.text((4, img.height - 14), "  ".join(note), fill=(255, 255, 0))
        images.append(img)

    # Hold the last frame a little longer so the outcome is readable.
    durations = [duration_ms] * len(images)
    durations[-1] = duration_ms * 4
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,
        loop=0,
        disposal=2,
    )
    return out_path


def plan_trajectory_figure(result, out_path: str | Path | None = None):
    """Top-down plot of every candidate trajectory, with the chosen plan bold.

    ``result`` is a :class:`aerojepa.sim.planner.PlanResult`. Returns the
    matplotlib Figure (and saves it if ``out_path`` is given).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aerojepa.viz.style import PALETTE, apply_style

    apply_style()
    positions = result.positions.cpu().numpy()  # (N, horizon, 3)
    costs = result.costs.cpu().numpy()
    best = result.best_index

    fig, (ax_xy, ax_cost) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    order = np.argsort(costs)[::-1]  # draw worst first so the best sits on top
    cmax = float(costs.max()) or 1.0
    for i in order:
        xy = positions[i]
        shade = 0.25 + 0.5 * (1.0 - costs[i] / cmax)
        ax_xy.plot(xy[:, 0], xy[:, 1], color=(0.6, 0.6, 0.6), alpha=shade, linewidth=0.8)
    bestxy = positions[best]
    ax_xy.plot(bestxy[:, 0], bestxy[:, 1], "-o", color=PALETTE["looped"], linewidth=2.2, label="chosen plan")
    ax_xy.scatter([0], [0], color="black", zorder=5, label="start")
    ax_xy.set_xlabel("x displacement")
    ax_xy.set_ylabel("y displacement")
    ax_xy.set_title("Candidate plans (top-down)")
    ax_xy.legend(loc="best", fontsize=8)
    ax_xy.set_aspect("equal", adjustable="datalim")

    ax_cost.hist(costs, bins=min(20, len(costs)), color=PALETTE["baseline"], alpha=0.85)
    ax_cost.axvline(costs[best], color=PALETTE["looped"], linewidth=2.0, label="chosen")
    ax_cost.set_xlabel("plan cost (lower is better)")
    ax_cost.set_ylabel("candidates")
    ax_cost.set_title("Cost distribution")
    ax_cost.legend(loc="best", fontsize=8)

    fig.tight_layout()
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
    return fig
