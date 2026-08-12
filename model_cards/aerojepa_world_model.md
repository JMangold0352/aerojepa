# Model card: AeroJEPA-WorldModel

## Overview

AeroJEPA-WorldModel is the **forward-prediction** variant: given the first frames
of a clip, it predicts the *latents* of the frames that follow. This is the
capability that supports predictive planning and obstacle anticipation. An
optional action-conditioned version is told the drone's 6-DoF motion, making it a
building block for model-based control.

- **Objective:** future-frame latent prediction (`objective: future`).
- **Configs:**
  [`configs/aerojepa_world_model.yaml`](../configs/aerojepa_world_model.yaml) and
  [`configs/aerojepa_action_conditioned.yaml`](../configs/aerojepa_action_conditioned.yaml).

## Architecture

Same encoder/teacher/predictor stack as
[AeroJEPA-Base](aerojepa_base.md), with these differences:

| Component | Setting |
| --- | --- |
| Objective | context = first `num_context_frames`; targets = remaining frames |
| Predictor recipe | `world_model: true` -> RMSNorm + SwiGLU + sandwich norm |
| Recurrence | looped, `max_loops=3`, learned exit gate |
| Action conditioning | optional; 6-DoF per-frame motion added to every token |

## Intended use

- **Rollout / anticipation:** forecast near-future scene structure in latent
  space.
- **Model-based planning:** with action conditioning, imagine the latent outcome
  of candidate maneuvers before executing them (see
  [`sim/planner.py`](../src/aerojepa/sim/planner.py) and the closed-loop stack in
  [aerojepa_action_wilds](aerojepa_action_wilds.md)).

## Out-of-scope / cautions

- Predicts latents, **not pixels** -- it does not render future frames.
- Closed-loop demos use a heuristic action map plus optional residual; research
  reference, not a flight controller.
- Synthetic-data numbers in this card; see `REPORT.md` and
  [`docs/EVAL_PROTOCOL.md`](../docs/EVAL_PROTOCOL.md) for real-flight and
  closed-loop evals.

## Performance

```bash
python scripts/train.py --config configs/aerojepa_world_model.yaml
python scripts/evaluate.py --checkpoint checkpoints/world_model/latest.pt
```

| Metric | Value | How |
| --- | --- | --- |
| Latent cosine (synthetic val) | **0.981** | `results/world_model_eval.json` |
| Rollout cosine @ horizon 1 | 0.974 | `results/world_model_eval.json` |
| Rollout cosine @ horizon 4 | **0.973** | flat curve (healthy) |
| Per-loop cosine (loops 1→3) | 0.87 → 0.96 → 0.98 | `results/world_model_eval.json` |
| Action-conditioned cosine | 0.980 (≈ unconditioned) | `results/action_conditioned_eval.json` |

Figures: [`visualizations/figures/world_model/`](../visualizations/figures/world_model/)

## Load and plan

```python
import torch
from aerojepa.eval import load_model
from aerojepa.sim import LatentPlanner

model, cfg = load_model("checkpoints/action_conditioned/latest.pt", torch.device("cpu"))
planner = LatentPlanner(model, torch.device("cpu"))
context = torch.rand(cfg["masking"]["num_context_frames"], 3, cfg["data"]["img_size"], cfg["data"]["img_size"])
best_actions = planner.plan(context, num_candidates=32)   # (num_temporal, 6)
```

## Lineage

Extends [looped-jepa](https://github.com/JMangold0352/looped-jepa) into a temporal
forward world model with optional action conditioning.
