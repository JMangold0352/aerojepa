# Cursor kickoff prompts

Copy-paste prompts for extending AeroJEPA in Cursor, one at a time. They are
written to preserve the structure, style, and quality bar of this codebase and
its parent, [looped-jepa](https://github.com/JMangold0352/looped-jepa). Each
prompt assumes the agent has read the relevant files first.

Style reminders to keep in every prompt:

- Match the existing code style: `from __future__ import annotations`, type hints,
  concise docstrings that explain *why*, config-driven behavior via `_base_`
  YAML inheritance.
- Keep the senior-researcher voice: motivate decisions, report trade-offs and
  negative results honestly, and keep explanations accessible to non-specialists.
- No emojis. Comments explain intent, never narrate the obvious.

---

## Prompt 1 - Ingest real UAV footage (Phase 2)

> Read `src/aerojepa/data/video_dataset.py`, `data/synthetic.py`, and
> `train.py`. Add a config option `data.source: video` path end-to-end: verify
> `VideoClipDataset` produces `(frames, actions)` matching the synthetic dataset,
> add a small config `configs/aerojepa_real.yaml` inheriting the base recipe with
> `data.source: video` and a `data_dir`, and document the expected folder layout
> in the README. Keep OpenCV an optional dependency. Add a test that skips
> gracefully when OpenCV or data is absent.

## Prompt 2 - MotionScape / AeroVerse loaders (Phase 2)

> Add a dataset module under `src/aerojepa/data/` for the target public UAV
> corpus. Follow the `VideoClipDataset` interface exactly (`(frames, actions)`),
> map the dataset's telemetry into our 6-DoF `ACTION_COLUMNS` convention in
> `telemetry.py`, and add a config that trains on it. Document licensing and the
> download step; do not hardcode credentials.

## Prompt 3 - Scale up training and run ablations (Phase 2)

> Using `scripts/run_ablations.py` as a template, run the baseline vs looped vs
> world-model variants for a real training budget. Record honest numbers
> (including any negative results) in `REPORT.md` and fill in the results table
> in the README and model cards. Do not overstate gains.

## Prompt 4 - Attention-over-time study (Phase 3)

> Extend `visualizations/inference.py` and `viz/plots.py` to render how predictor
> attention evolves across refinement loops and across frames on real footage.
> Add the figure to `generate_all_figures.py` and describe how to read it in
> `visualizations/README.md`.

## Prompt 5 - Close the planning loop (Phase 4)

> Read `src/aerojepa/sim/planner.py` and `sim/simulators.py`. Replace the
> placeholder `smoothness_cost` with a task cost (start with a learned
> collision-risk head trained on synthetic obstacle labels). Evaluate the
> `LatentPlanner` in a PyFlyt hover/obstacle task and report a task-success
> metric. Keep the simulator dependency optional.

## Prompt 6 - Model cards with real numbers (Phase 5)

> Once real training numbers exist, update `model_cards/aerojepa_base.md` and
> `model_cards/aerojepa_world_model.md` with measured performance, limitations,
> and load-and-run snippets. Keep the tone factual and the claims defensible.
