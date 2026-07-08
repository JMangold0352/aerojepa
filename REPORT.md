# AeroJEPA experiment log

A running, honest record of what has been tried and what it showed.

## Phase 1 - Architecture & synthetic benchmark — **complete**

**Goal.** Video JEPA world model + reproducible synthetic benchmark.

**Setup.** 1024 train / 128 val synthetic clips, 64×64 RGB, 8 frames, 100 epochs,
MPS (Apple Silicon), configs in `configs/aerojepa_*.yaml`.

**Results (latent cosine, held-out synthetic clips):**

| Model | Cosine | Notes |
| --- | ---: | --- |
| baseline (feed-forward) | 0.954 | masked objective |
| looped (+ sandwich RMSNorm, exit gate) | **0.961** | **+0.7 pp** vs baseline |
| world_model (future frames) | **0.981** | rollout flat ~0.97 over 4-frame horizon |
| action_conditioned | 0.980 | 6-DoF telemetry; no gain on synthetic |

**Takeaways:**

- Recurrence + sandwich RMSNorm **does** help on video (opposite of naive looping
  hurting on CIFAR in the parent project — normalization was the key there too).
- Per-loop refinement on the world model is clear: cosine **0.87 → 0.96 → 0.98**
  over 3 loops; exit gate averages **1.75** steps.
- Action conditioning did not separate on procedural synthetic data; real Tello
  telemetry is the honest test (Phase 2).

Artifacts: `checkpoints/{baseline,looped,world_model,action_conditioned}/`,
`results/*_eval.json`, figures in `visualizations/figures/`.

---

## Phase 2 - Real data & transfer — **in progress**

**Goal.** Train on real UAV footage; measure sim-to-real gap; update model cards.

**Completed:**

- The Wilds Drones ingested (15 Parrot clips) → `data/flights_128/`.
- Fast fine-tune (`real_finetune_fast`): latent cosine **0.994**, rollout **0.984**, gap **+0.019**.
- Tello workflow + transfer comparison report scaffolded.

**Next steps:**

1. Capture Tello clips (`scripts/tello_workflow.sh`).
2. Closed-loop PyFlyt planner eval.
3. Re-run ablation suite at 100 epochs for publication table (`--mode full`).

---

## Transfer curve — sim-to-real vs data volume

**Goal.** Quantify how quickly synthetic pretraining transfers as real clip count grows.

**Setup.**

- Init: `checkpoints/world_model/latest.pt` (synthetic, cosine 0.981).
- Data: `data/flights_128/` (15 Wilds Parrot clips, 128×128).
- Eval: **3 clips held out** (`wilds_012`–`wilds_014`); never used in training.
- Train subsets: **1, 5, 12** clips (request size 15 → all clips outside holdout).
- Fine-tune: 5 epochs per point, LR 5e-5 (`configs/aerojepa_transfer_curve.yaml`).

**Command.**

```bash
python scripts/run_transfer_curve.py --device mps
# → results/transfer_curve/summary.json + transfer_curve.png
```

**Results** — [`results/transfer_curve/summary.json`](results/transfer_curve/summary.json) · figure: [`transfer_curve.png`](results/transfer_curve/transfer_curve.png)

Eval holdout: `wilds_012`–`wilds_014` (3 clips, never trained on). Init: synthetic `world_model`. **5 epochs** per point, LR 5e-5.

| Train clips | Real latent cosine ↑ | Sim-to-real gap ↓ | Rollout @ h=4 ↑ |
| --- | ---: | ---: | ---: |
| **0** (synthetic only) | 0.990 | **−0.009** | 0.984 |
| **1** | 0.993 | +0.003 | **0.988** |
| **5** | **0.994** | +0.002 | **0.988** |
| **12** (max pool; ≈15 corpus) | **0.994** | +0.002 | 0.987 |

**Takeaways:**

- **Most transfer happens with one clip.** Real latent cosine jumps 0.990 → 0.993 and rollout improves after a single 30s Parrot session worth of data; 5–12 clips add only ~0.001 cosine.
- **Baseline gap is negative** on this holdout (−0.009): without fine-tuning, held-out real clips are *easier* than the synthetic reference embedded in the checkpoint — the gap metric is relative to the model's own synthetic val recipe, not absolute quality. After fine-tune the gap turns slightly positive as the synthetic branch rises faster than real.
- **Rollout tracks latent quality** — flat ~0.987–0.988 after any real fine-tune; no sign of horizon collapse at h=4.
- **Diminishing returns by clip 5** — supports the portfolio story: synthetic pretrain + a handful of real flights, not a massive corpus, for this model size and recipe.
- Compare upper bound: `real_finetune_fast` (10 epochs, all 15 clips, no fixed holdout) reaches 0.994 val / 0.974 held-out real in the full-corpus eval — stricter split, different protocol.

Artifacts: `results/transfer_curve/`, `checkpoints/transfer_curve/n{1,5,12}/`,
figure at `results/transfer_curve/transfer_curve.png` (+ PDF).

---

## Ablation suite (20-epoch quick mode) — **complete**

**Goal.** Isolate feed-forward vs loop count vs future objective on the same synthetic recipe.

**Command.**

```bash
python scripts/run_ablations.py --mode quick
python visualizations/compare_ablations.py
```

**Results (latent cosine, held-out synthetic, 20 epochs):**

| Variant | Objective | Cosine | Rollout h=4 | Per-loop gain |
| --- | --- | ---: | ---: | --- |
| baseline | masked | 0.993 | 0.990 | — |
| loops_2 | masked | 0.993 | 0.989 | +0.025 |
| loops_3 | masked | 0.993 | 0.990 | +0.112 |
| world_model | future | **0.994** | **0.992** | +0.124 |

**Takeaways.**

- At 20 epochs, headline latent cosine does **not** separate masked variants (all ~0.993).
- Per-loop refinement **does** separate: world model rises **0.87 → 0.98 → 0.99** over 3 loops.
- Future objective wins on both latent cosine and rollout at this budget.
- For publication-quality ablation numbers, re-run with `--mode full` (100 epochs).

Artifacts: `results/ablations/summary.json`, per-variant JSON, figures in
`visualizations/figures/ablations/` (bar charts, per-loop gain, rollout panel + GIF).

---

## 2026-07-03 - Synthetic training complete

- Configs: baseline, looped, world_model, action_conditioned (100 ep each).
- Hardware: MPS, ~3.2M params (world model ~5.1M with SwiGLU recipe).
- Best masked: looped 0.961. Best future: world_model 0.981.
- Rollout horizon 4 still ~0.97 for world model (healthy flat curve).
