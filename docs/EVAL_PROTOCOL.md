# Eval protocol

One place for how numbers in `README.md`, `REPORT.md`, and model cards are
produced. If a metric disagrees with another table, check which protocol it used.

## Resolution (64 vs 128)

| Layer | Resolution | Notes |
| --- | ---: | --- |
| On-disk preprocess (`data/flights_128/`) | **128×128** | Square crop + resize in `scripts/preprocess_real.py` |
| Model input (`data.img_size` in configs) | **64×64** | Inherited from `configs/aerojepa_synth_base.yaml` |
| Closed-loop render | **64×64** | Matches checkpoint `img_size` |

Training and eval **always** resize clips to `cfg["data"]["img_size"]` (64 today).
Saying a checkpoint was “trained on 128 footage” means the **source files** are
128 on disk; the network still sees 64×64. Do not compare a true 128-input model
to these numbers without a separate recipe (patch grid and positional embeddings
change).

## World-model metrics

Primary metric: **latent cosine** between predictor and EMA teacher on held-out
clips (higher is better). Secondary: multi-step **rollout cosine** at horizons
1–4, and per-loop refinement when the predictor is looped.

### A. Synthetic (in-distribution)

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/<name>/latest.pt \
  --max-batches 8
# → results/<name>_eval.json
```

Uses the synthetic val set from the checkpoint config (`num_val`, seed).
Default `max_batches=8` is the published budget — keep it when regenerating
tables.

### B. Sim-to-real gap

```bash
python scripts/evaluate_real.py \
  --checkpoint checkpoints/<name>/latest.pt \
  --data-dir data/flights_128 \
  --window-mode uniform \
  --max-batches 8 \
  --out results/<name>_real_eval.json
```

Same metric code on synthetic vs real. Gap = synthetic cosine − real cosine
(positive ⇒ real is harder). **Do not** quote training-time val cosine from
`val_fraction` as this gap — that split is random whole-video holdout and is not
the same as this script.

Published comparisons for Wilds fine-tunes use `--data-dir data/flights_128`.

### C. Transfer curve (fixed holdout)

```bash
python scripts/run_transfer_curve.py --device mps
# → results/transfer_curve/summary.json
```

Fixed eval holdout: last **3** clips (`wilds_012`–`wilds_014`). Train subsets
1 / 5 / 12 from the remaining pool. Do not mix these numbers with protocol B
without labeling the split.

## Closed-loop (PyFlyt)

Requires PyFlyt; run **outside** the Cursor sandbox (native physics can segfault
under sandboxing).

### Default demo stack

```bash
python scripts/run_closed_loop_demo.py \
  --checkpoint checkpoints/action_conditioned_wilds/latest.pt \
  --residual-checkpoint checkpoints/action_residual_wilds/best.pt \
  --planner gradient \
  --latent-smooth 0.05 \
  --task hover
```

Tasks: `hover`, `waypoint`, `recover`, `wind_gust`, `aggressive_turn`.

### Multi-seed full-stack table

```bash
python scripts/compare_full_stack.py \
  --checkpoint checkpoints/action_conditioned_wilds/latest.pt \
  --residual-checkpoint checkpoints/action_residual_wilds/best.pt \
  --planner gradient \
  --latent-smooth 0.05 \
  --seeds 0,1,2 \
  --include-hard-turn \
  --out visualizations/closed_loop/full_stack_compare_wilds.json
```

Report **success rate** and **mean max XY drift** from the `aggregate` block.
Seed lists and latent-smooth must be stated next to the table.

Stress-only baseline (heuristic map, no residual): see
`visualizations/closed_loop/stress/BREAKING_POINTS.md`.

## Ablations

```bash
python scripts/run_ablations.py --mode quick   # 20 epochs — iteration
python scripts/run_ablations.py --mode full    # 100 epochs — publication
python visualizations/compare_ablations.py
# → results/ablations/summary.json
```

Always label tables with `quick` vs `full`. Do not mix epoch budgets in one
column.

## Which checkpoint for which claim

| Claim | Checkpoint | Protocol |
| --- | --- | --- |
| Best **unconditioned** Wilds representation | `checkpoints/real_finetune_fast/latest.pt` | B (`*_real_eval.json`) |
| Best **closed-loop** stack | `action_conditioned_wilds` + `action_residual_wilds` | Closed-loop multi-seed |
| Synthetic architecture comparison | `baseline` / `looped` / `world_model` / `action_conditioned` | A |
| Publication ablations | `results/ablations/` after `--mode full` | Ablations |

Action-conditioned Wilds is weaker than `real_finetune_fast` on protocol B real
cosine (~0.957 vs ~0.974) but is required for gradient planning + residual.
Keep those two stories separate in public tables.
