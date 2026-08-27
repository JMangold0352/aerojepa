# AeroJEPA experiment log

What was tried and what it showed. Metric protocols:
[`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md).

**Related work.** Closest paper: SkyJEPA (Rao et al., arXiv:2606.23444, 2026) -
state-history JEPA + physics-inspired prober + outdoor MPPI. AeroJEPA is a
*video*-JEPA on egocentric clips; do not compare AeroProber short-horizon sim RMSE
to SkyJEPA outdoor tracking. See README Related work for BibTeX.

## Architecture & synthetic benchmark - **complete**

**Goal.** Video JEPA world model + reproducible synthetic benchmark.

**Setup.** 1024 train / 128 val synthetic clips, **64×64** RGB model input, 8 frames,
100 epochs, MPS (Apple Silicon), configs in `configs/aerojepa_*.yaml`.

**Results (latent cosine, held-out synthetic clips):**

| Model | Cosine | Notes |
| --- | ---: | --- |
| baseline (feed-forward) | 0.954 | masked objective |
| looped (+ sandwich RMSNorm, exit gate) | **0.961** | **+0.7 pp** vs baseline |
| world_model (future frames) | **0.981** | rollout flat ~0.97 over 4-frame horizon |
| action_conditioned | 0.980 | 6-DoF telemetry; no gain on synthetic |

**Takeaways:**

- Recurrence + sandwich RMSNorm **does** help on video (opposite of naive looping
  hurting on CIFAR in the parent project - normalization was the key there too).
- Per-loop refinement on the world model is clear: cosine **0.87 → 0.96 → 0.98**
  over 3 loops; exit gate averages **1.75** steps.
- Action conditioning did not separate on procedural synthetic data; real Tello
  telemetry is the better test.

Artifacts: `checkpoints/{baseline,looped,world_model,action_conditioned}/`,
`results/*_eval.json`, figures in `visualizations/figures/`.

---

## Real data & transfer - **complete** (Wilds); Tello pending

**Goal.** Train on real UAV footage; measure sim-to-real gap; support closed-loop.

**Completed:**

- The Wilds Drones ingested (15 Parrot clips) → `data/flights_128/` (128 on disk;
  model still trains at **64×64**).
- Unconditioned fine-tune (`real_finetune_fast`): protocol-B real cosine **0.974**,
  synthetic **0.994**, gap **+0.019**
  ([`results/real_finetune_fast_eval.json`](results/real_finetune_fast_eval.json)).
- Action-conditioned Wilds fine-tune (`action_conditioned_wilds`): protocol-B real
  cosine **0.957**, gap **+0.037**
  ([`results/action_conditioned_wilds_real_eval.json`](results/action_conditioned_wilds_real_eval.json)).
  Weaker on representation metrics; required for gradient planning + residual.
- Transfer curve (fixed 3-clip holdout): most of the gain with 1 clip; see below.
- Closed-loop stack + residual (next section).

**Still open:**

1. Capture Tello clips (`scripts/tello_workflow.sh`) - no footage on disk yet.
2. Better action-Wilds FT than v1 - **v2 failed** (protocol-B real cosine 0.915 vs v1 0.957); keep v1 as default.
3. AeroProber open physics questions - [`research/prober/note.md`](research/prober/note.md) §9.

---

## Closed-loop PyFlyt - **working** (research demo)

**Goal.** Use the world model inside a receding-horizon loop with a residual
action correction.

**Default stack (v1):**

```bash
python scripts/run_closed_loop_demo.py \
  --checkpoint checkpoints/action_conditioned_wilds/latest.pt \
  --residual-checkpoint checkpoints/action_residual_wilds/best.pt \
  --planner gradient --latent-smooth 0.05
```

**Hard-task result.** Success vs difficulty, **10 seeds**, v1 stack -
[`scripts/run_hard_pyflyt_suite.py`](scripts/run_hard_pyflyt_suite.py) →
[`visualizations/closed_loop/stress_suite.json`](visualizations/closed_loop/stress_suite.json)
(+ [`stress_suite_success.png`](visualizations/closed_loop/stress_suite_success.png)):

| Sweep | Outcome |
| --- | --- |
| Wind 0-4 m/s | ~100% |
| Recover delay / hover kick | ~100% |
| L-turn scale ×0.5-1.0 | ~100% |
| **L-turn scale ×1.25** | **0%** (cliff) |

Easy tasks saturate; the hard L-turn does not.

**Easy-task saturation (3 seeds).** Older table (`latent_smooth=0.05`) -
[`full_stack_compare_wilds.json`](visualizations/closed_loop/full_stack_compare_wilds.json)
/ bake-off [`bakeoff_v1_v2_summary.json`](visualizations/closed_loop/bakeoff_v1_v2_summary.json):

| Task | Full-stack success | Mean max XY (m) |
| --- | ---: | ---: |
| wind_gust (2 m/s) | 100% | 0.35 |
| aggressive_turn (soft) | 100% | 1.38 |
| aggressive_turn_hard (0.8 m) | 100% | 0.97 |
| recover | 100% | 0.92 |
| hover | 100% | 0.08 |

**v2 continuation** (`action_conditioned_wilds_v2` + residual_v2) improved residual
val MSE but dropped protocol-B real cosine (0.915 vs 0.957) and failed soft L-turn
on 2/3 seeds. Not adopted as default.

Recover survival uses adaptive braking after the kick (shared across policies).
Older stress notes: [`visualizations/closed_loop/stress/BREAKING_POINTS.md`](visualizations/closed_loop/stress/BREAKING_POINTS.md).

**Residual heads:**

| Checkpoint | Mix | Notes |
| --- | --- | --- |
| `checkpoints/action_residual_wind/` | wind | early wind-only recipe |
| `checkpoints/action_residual_multi/` | wind+kick+turn | synthetic WM |
| `checkpoints/action_residual_wilds/` | wind+kick+turn | Wilds action WM (**preferred**) |

---

## Transfer curve - sim-to-real vs data volume

**Goal.** Quantify how quickly synthetic pretraining transfers as real clip count grows.

**Setup.**

- Init: `checkpoints/world_model/latest.pt` (synthetic, cosine 0.981).
- Data: `data/flights_128/` (15 Wilds Parrot clips, 128 on disk → 64 model input).
- Eval: **3 clips held out** (`wilds_012`-`wilds_014`); never used in training.
- Train subsets: **1, 5, 12** clips (request size 15 → all clips outside holdout).
- Fine-tune: 5 epochs per point, LR 5e-5 (`configs/aerojepa_transfer_curve.yaml`).

**Command.**

```bash
python scripts/run_transfer_curve.py --device mps
# → results/transfer_curve/summary.json + transfer_curve.png
```

**Results** - [`results/transfer_curve/summary.json`](results/transfer_curve/summary.json) · figure: [`transfer_curve.png`](results/transfer_curve/transfer_curve.png)

Eval holdout: `wilds_012`-`wilds_014` (3 clips, never trained on). Init: synthetic `world_model`. **5 epochs** per point, LR 5e-5.

| Train clips | Real latent cosine ↑ | Sim-to-real gap ↓ | Rollout @ h=4 ↑ |
| --- | ---: | ---: | ---: |
| **0** (synthetic only) | 0.990 | **−0.009** | 0.984 |
| **1** | 0.993 | +0.003 | **0.988** |
| **5** | **0.994** | +0.002 | **0.988** |
| **12** (max pool; ≈15 corpus) | **0.994** | +0.002 | 0.987 |

**Takeaways:**

- Synthetic-only already reaches ~0.990 real cosine on this 3-clip holdout, and
  action counterfactuals on `action_conditioned_wilds` are flat
  (true/zero/shuffle ≈ 0.994); the latent task may be easy or partly collapsed.
- **Most transfer happens with one clip.** Real latent cosine jumps 0.990 → 0.993 and rollout improves after a single 30s Parrot session worth of data; 5-12 clips add only ~0.001 cosine.
- **Baseline gap is negative** on this holdout (−0.009): without fine-tuning, held-out real clips are *easier* than the synthetic reference embedded in the checkpoint - the gap metric is relative to the model's own synthetic val recipe, not absolute quality. After fine-tune the gap turns slightly positive as the synthetic branch rises faster than real.
- **Rollout tracks latent quality** - flat ~0.987-0.988 after any real fine-tune; no sign of horizon collapse at h=4.
- **Diminishing returns by clip 5** - for this model size, synthetic pretrain plus a handful of real flights carries most of the transfer; more clips add little on this holdout.
- Compare protocol B upper bound: `real_finetune_fast` reaches **0.974** real cosine on the full-corpus `evaluate_real.py` split - different protocol than this fixed holdout.

Artifacts: `results/transfer_curve/`, `checkpoints/transfer_curve/n{1,5,12}/`,
figure at `results/transfer_curve/transfer_curve.png` (+ PDF).

---

## Ablation suite (100-epoch full mode) - **complete**

**Goal.** Isolate feed-forward vs loop count vs future objective on the same synthetic recipe.

**Command.**

```bash
python scripts/run_ablations.py --mode full
python visualizations/compare_ablations.py
```

**Results (latent cosine, held-out synthetic, 100 epochs):**

| Variant | Objective | Cosine | Rollout h=4 | Per-loop cosine |
| --- | --- | ---: | ---: | --- |
| baseline | masked | 0.953 | 0.917 | - |
| loops_2 | masked | 0.960 | 0.925 | 0.941 → 0.960 |
| loops_3 | masked | 0.963 | 0.928 | 0.841 → 0.962 |
| world_model | future | **0.981** | **0.973** | 0.869 → 0.960 → 0.981 |

**Takeaways.**

- At full budget, recurrence separates: loops_3 **+0.95 pp** over baseline; world model **+2.8 pp**.
- Future objective wins latent cosine and keeps rollout flat ~0.973 over h=4.
- Per-loop refinement on the world model is clear (0.87 → 0.98); exit gate ~**1.75** steps.
- Matches the original synthetic suite headlines in §Architecture (within float noise).

Quick-mode (20 ep) numbers are not used in publication tables; they inflated all
variants to ~0.993 without separating architecture.

Artifacts: [`results/ablations/summary.json`](results/ablations/summary.json), figures in
`visualizations/figures/ablations/`.

---

## 2026-07-03 - Synthetic training complete

- Configs: baseline, looped, world_model, action_conditioned (100 ep each).
- Hardware: MPS, ~3.2M params (world model ~5.1M with SwiGLU recipe).
- Best masked: looped 0.961. Best future: world_model 0.981.
- Rollout horizon 4 still ~0.97 for world model (healthy flat curve).
