from __future__ import annotations

import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aerojepa.sim.rollout_demo import plan_and_render
from aerojepa.viz.style import PALETTE, apply_style
from demo.inference import DemoModel

INTRO = """
# AeroJEPA - interactive world-model demo

AeroJEPA watches a few frames of drone video and predicts what comes next -- not
as pixels, but in a compact **latent** space (the same idea behind JEPA world
models). This demo lets you:

1. **Generate a flight clip** with the built-in synthetic drone camera.
2. Choose how many frames the model sees before it has to **predict the future**.
3. Dial the **number of refinement loops** and watch prediction quality change.

*Cosine similarity* (0 to 1) measures how close the predicted latent is to the
truth -- higher is better.
"""

PLANNER_INTRO = """
## Latent-space planner

A world model can **plan by imagining**: sample many candidate action plans, roll
each one forward in latent space, score the imagined futures, and execute the
best one -- all without touching the real drone. Below, AeroJEPA plans over a
short synthetic flight and then *executes* the winning plan in the camera
simulator so you can watch what it chose to do.

- **Hover** = hold position. **Waypoint** = fly to a goal. **Smoothness** =
  keep the imagined scene coherent.
- Blue banner = observed context; orange banner = the planned rollout.

Action-conditioned checkpoints score candidates with the model's own latents;
plain world models fall back to a kinematic cost (still great for hover/waypoint).
"""


def _clip_gallery(clip) -> list[np.ndarray]:
    return [
        (clip[t].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype("uint8")
        for t in range(clip.shape[0])
    ]


def _rollout_figure(quality: dict):
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(quality["horizon"], quality["cosine"], "-o", color=PALETTE["looped"])
    ax.set_xlabel("frames into the future")
    ax.set_ylabel("cosine to true latent")
    ax.set_title("Prediction quality vs how far ahead")
    ax.set_ylim(min(0.0, min(quality["cosine"]) - 0.05), 1.02)
    fig.tight_layout()
    return fig


def _loop_figure(loop: dict | None):
    apply_style()
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    if loop is None:
        ax.text(0.5, 0.5, "This model is feed-forward\n(no refinement loops).", ha="center", va="center")
        ax.axis("off")
        return fig
    loops = list(range(1, len(loop["per_loop_cosine"]) + 1))
    ax.plot(loops, loop["per_loop_cosine"], "-o", color=PALETTE["baseline"])
    ax.set_xticks(loops)
    ax.set_xlabel("refinement loop")
    ax.set_ylabel("cosine to true latent")
    ax.set_title("Does thinking longer help?")
    fig.tight_layout()
    return fig


def launch_demo(checkpoint: str | None = None, share: bool = False) -> None:
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("The demo needs gradio. Install with `pip install gradio`.") from exc

    demo_model = DemoModel(checkpoint=checkpoint)
    status = "trained checkpoint" if demo_model.trained else "UNTRAINED model (smoke config)"
    max_t = demo_model.num_temporal

    def run(seed: int, context_frames: int, max_loops: int):
        clip = demo_model.sample_clip(int(seed))
        quality = demo_model.future_quality(clip, int(context_frames), int(max_loops))
        loop = demo_model.loop_stats(clip, int(max_loops))

        avg_cos = sum(quality["cosine"]) / max(1, len(quality["cosine"]))
        lines = [
            f"**Model:** {status}",
            f"**Sees:** first {quality['context_frames']} of {max_t} frames",
            f"**Average future cosine:** {avg_cos:.3f}",
        ]
        if loop and loop.get("expected_loops") is not None:
            lines.append(f"**Expected refinement loops:** {loop['expected_loops']:.2f}")
        return _clip_gallery(clip), _rollout_figure(quality), _loop_figure(loop), "\n\n".join(lines)

    def run_planner(seed: int, task: str, context_frames: int, num_candidates: int,
                    action_scale: float, goal_x: float):
        gif_path = tempfile.mkstemp(suffix=".gif")[1]
        out = plan_and_render(
            demo_model.model,
            demo_model.device,
            img_size=demo_model.cfg["data"]["img_size"],
            in_chans=demo_model.cfg["data"].get("in_chans", 3),
            num_obstacles=demo_model.cfg["data"].get("num_obstacles", 5),
            seed=int(seed),
            task=task,
            context_frames=int(context_frames),
            num_candidates=int(num_candidates),
            action_scale=float(action_scale),
            goal=(float(goal_x), 0.0, 0.0) if task == "waypoint" else None,
            out_gif=gif_path,
            make_figure=True,
        )
        r = out.result
        mode = "model latents" if r.action_conditioned else "kinematic cost (plain world model)"
        summary = "\n\n".join([
            f"**Model:** {status}",
            f"**Task:** {task}  |  **scored by:** {mode}",
            f"**Planned:** {r.horizon} steps from {r.context_frames} context frames "
            f"over {int(num_candidates)} candidates",
            f"**Best plan cost:** {float(r.costs[r.best_index]):.4f}",
            f"**Imagined coherence:** {r.coherence:.3f}",
        ])
        return str(out.gif_path), out.figure, summary

    with gr.Blocks(title="AeroJEPA") as blocks:
        gr.Markdown(INTRO)
        with gr.Tabs():
            with gr.Tab("Predict the future"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed = gr.Slider(0, 999, value=7, step=1, label="Flight clip (seed)")
                        context = gr.Slider(1, max_t - 1, value=max(1, max_t // 2), step=1, label="Frames the model sees")
                        loops = gr.Slider(1, 4, value=2, step=1, label="Refinement loops")
                        go = gr.Button("Predict the future", variant="primary")
                    with gr.Column(scale=2):
                        gallery = gr.Gallery(label="Flight clip (time ->)", columns=max_t, height=160)
                        stats = gr.Markdown()
                with gr.Row():
                    rollout_plot = gr.Plot(label="Rollout quality")
                    loop_plot = gr.Plot(label="Refinement")
                go.click(run, [seed, context, loops], [gallery, rollout_plot, loop_plot, stats])

            with gr.Tab("Run Latent Planner"):
                gr.Markdown(PLANNER_INTRO)
                with gr.Row():
                    with gr.Column(scale=1):
                        p_seed = gr.Slider(0, 999, value=7, step=1, label="Flight clip (seed)")
                        p_task = gr.Radio(["hover", "waypoint", "smoothness"], value="hover", label="Task")
                        p_context = gr.Slider(1, max_t - 1, value=max(1, max_t // 2), step=1, label="Context frames")
                        p_cands = gr.Slider(8, 256, value=96, step=8, label="Candidate plans")
                        p_scale = gr.Slider(0.01, 0.2, value=0.06, step=0.01, label="Action magnitude")
                        p_goal = gr.Slider(-0.4, 0.4, value=0.3, step=0.05, label="Waypoint goal (x)")
                        p_go = gr.Button("Run Latent Planner", variant="primary")
                    with gr.Column(scale=2):
                        plan_gif = gr.Image(label="Observed context -> planned rollout", type="filepath")
                        plan_stats = gr.Markdown()
                plan_traj = gr.Plot(label="Candidate plans and cost distribution")
                p_go.click(
                    run_planner,
                    [p_seed, p_task, p_context, p_cands, p_scale, p_goal],
                    [plan_gif, plan_traj, plan_stats],
                )

    blocks.launch(share=share)
