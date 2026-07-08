# AeroJEPA — Project Status & Decision Brief

**Author:** John Mangold  
**Date:** July 3, 2026  
**Repo:** `~/Projects/aerojepa`  
**Parent project:** [looped-jepa](https://github.com/JMangold0352/looped-jepa)

Read this once to understand what exists, what it showed, and what you can do next.

---

## In one sentence

We built **AeroJEPA** — a recurrent video world model for drones that learns to predict what happens next in *latent space* (not pixels), extending your Looped-JEPA predictor into time — and we trained four model variants on a built-in synthetic benchmark with strong numbers and a full eval/demo pipeline.

---

## What AeroJEPA is

| Idea | Plain language |
| --- | --- |
| **World model** | Predicts the near future so a drone can anticipate, not just react. |
| **JEPA** | Learns by predicting *meaning* (latents), not reconstructing every pixel. |
| **Looped predictor** | “Thinks longer” by reusing the same small network several times; a learned exit gate stops early when the scene is easy. |
| **Synthetic benchmark** | Procedural drone clips (moving camera over terrain + obstacles) so everything trains with **zero downloads**. |

**Defense / autonomy relevance:** compact models, label-efficient self-supervision, interpretable per-loop attention, adaptive compute, and a path to model-based planning in latent space.

---

## Timeline (this session)

| When | What happened |
| --- | --- |
| Morning | Full repo scaffolded from plan: models, data, training, eval, viz, demo, docs, tests. |
| ~11:09 | All-day training started (baseline → looped → world_model → action_conditioned). |
| ~11:24 | **Baseline** finished (100 epochs, MPS). |
| ~11:39 | **Looped** finished. |
| ~12:02 | **World model** finished. |
| ~12:26 | **Action-conditioned** finished. Total ~1 hr for last three models. |
| Afternoon | Full eval pass, figures, README/REPORT/model cards updated with real numbers. |
| Afternoon | Phase 2 started: real-data config, `data/README`, Tello capture script. |
| Afternoon | 20-epoch ablation suite **finished** → `results/ablations/summary.json`. |

---

## What was built (repository map)

```
aerojepa/
├── src/aerojepa/          Core library
│   ├── models/            ViT encoder, video predictor, looped predictor, AeroJEPA
│   ├── data/              Synthetic generator, real-video loader, Tello stub, telemetry
│   ├── masking.py         Masked + future-frame objectives
│   ├── train.py           Training loop (AdamW, EMA, exit-entropy)
│   ├── eval/              Latent quality, rollout, loop analysis, compare
│   ├── viz/               Publication figure helpers
│   └── sim/               Latent planner + optional PyFlyt/pybullet hooks
├── configs/               baseline, looped, world_model, action_conditioned, real, smoke
├── scripts/               train, evaluate, compare, ablations, visualize, demo, verify
├── checkpoints/           Four trained models (~49–51 MB each)
├── results/               JSON metrics + comparison
├── visualizations/        Figures from trained checkpoints
├── demo/ + app.py         Gradio interactive demo
├── model_cards/           Professional per-model docs
├── docs/                  Technical report, roadmap, Cursor kickoff prompts
├── tests/                 13 pytest tests (all passed)
└── README.md              Main entry point (polished, defense-framed)
```

**Key design choices**

- **~3.2M params** (baseline/looped) / **~5.1M** (world model with SwiGLU recipe) — under your 50M budget.
- **Factorized space-time positions** — cheap, readable temporal encoding.
- **Two objectives, one network:** swap mask collator only (`masked` vs `future`).
- **Sandwich RMSNorm + exit gate** on looped variants (lesson from looped-jepa).

---

## Trained models (synthetic benchmark, 100 epochs, MPS)

| Model | Config | Objective | Latent cosine | Rollout @ 4 frames | Expected loops | Checkpoint |
| --- | --- | --- | ---: | ---: | ---: | --- |
| **baseline** | `aerojepa_baseline.yaml` | masked patches | **0.954** | 0.917 | — | `checkpoints/baseline/latest.pt` |
| **looped** | `aerojepa_looped.yaml` | masked + recurrence | **0.961** | 0.925 | 1.50 | `checkpoints/looped/latest.pt` |
| **world_model** | `aerojepa_world_model.yaml` | predict future frames | **0.981** | 0.973 | 1.75 | `checkpoints/world_model/latest.pt` |
| **action_conditioned** | `aerojepa_action_conditioned.yaml` | future + 6-DoF motion | **0.980** | 0.975 | 1.75 | `checkpoints/action_conditioned/latest.pt` |

**Headline comparison (baseline vs looped):** **+0.007** latent cosine (0.959 vs 0.951 on a fresh compare run).

**World-model refinement (per-loop cosine):** 0.87 → 0.96 → **0.98** over 3 loops. Rollout stays **flat ~0.97** across horizon — a healthy world model signature on synthetic data.

**Action conditioning:** essentially tied with unconditioned world model on synthetic data (0.980 vs 0.981). Real telemetry is the honest test.

Full JSON: `results/baseline_eval.json`, `looped_eval.json`, `world_model_eval.json`, `action_conditioned_eval.json`, `comparison.json`.

### 20-epoch ablation sweep (quick signal)

| Variant | Objective | Latent cosine | Smooth L1 |
| --- | --- | ---: | ---: |
| baseline | masked | 0.993 | 0.0069 |
| loops_2 | masked | 0.993 | 0.0075 |
| loops_3 | masked | 0.993 | 0.0071 |
| world_model | future | **0.994** | **0.0059** |

All variants cluster near 0.99 on synthetic data at 20 epochs — the **future objective** edges out on both metrics. Full JSON: `results/ablations/summary.json`.

---

## Figures & demos

**Publication figures (from real checkpoints, not smoke):**

- `visualizations/figures/looped/` — clip, rollout, per-loop cosine, exit distribution, latent trajectory, attention
- `visualizations/figures/world_model/` — same suite for the forward world model

**Regenerate:**

```bash
python scripts/visualize.py --checkpoint checkpoints/looped/latest.pt --out-dir visualizations/figures/looped
python scripts/visualize.py --checkpoint checkpoints/world_model/latest.pt --out-dir visualizations/figures/world_model
```

**Interactive demo:**

```bash
pip install gradio   # if needed
python app.py --checkpoint checkpoints/world_model/latest.pt
```

Opens at http://127.0.0.1:7860 — generate a synthetic flight, choose context frames and loop count, see rollout quality live.

---

## Verification (what we proved works)

| Check | Result |
| --- | --- |
| `python scripts/verify_install.py` | Pass |
| `pytest` (38 tests) | All pass |
| Smoke train (1 epoch) | Pass |
| Full synthetic training (4 models) | Pass |
| Evaluate + compare + visualize | Pass |

---

## Phase status

### Phase 1 — Architecture & synthetic benchmark — **COMPLETE**

- Video JEPA + recurrent predictor + exit gate
- Synthetic drone generator (no external data)
- Training, eval, ablations harness, figures, Gradio demo
- Four trained checkpoints with measured metrics
- Documentation updated with real numbers

### Phase 2 — Real data & transfer — **PIPELINE COMPLETE**

**Done:**

- `configs/aerojepa_real.yaml` + `aerojepa_finetune.yaml` — train on `data/flights/`
- `data/README.md` — folder layout, Tello safety, preprocess workflow
- `scripts/capture_tello.py` — robust record-only Tello capture (Prompt 6)
- `scripts/preprocess_real.py` — dataset doctor + standardize
- `--init-checkpoint` + `--resume` on `scripts/train.py` (Prompt 7)
- `scripts/launch_training.sh` — reliable long runs from Terminal.app
- `scripts/evaluate_real.py` + `evaluate_all.sh` real-data gap
- **38 tests** passing

**Not done yet:**

- No real drone footage collected or fine-tuned on
- No MotionScape / AeroVerse loaders
- 100-epoch ablation suite (full mode) not run yet

### Phases 3–5 — Not started

- Hosted demo (Hugging Face Spaces)
- Attention studies on real footage
- PyFlyt closed-loop planning
- Paper-ready release / GitHub public push

---

## Important lessons from training today

1. **Cursor background `nohup` jobs died** when the shell session ended (~batch 21, no error). Training that survived ran in a **Cursor background shell** attached to `train.py` directly.
2. **For overnight runs**, use **Terminal.app**:
   ```bash
   cd ~/Projects/aerojepa && ./scripts/launch_in_terminal.sh
   ```
3. **Synthetic numbers are strong** — but defense stakeholders will ask for **real footage**. That is the critical credibility step.

---

## Commands cheat sheet

```bash
cd ~/Projects/aerojepa
source .venv/bin/activate

# Re-evaluate everything
./scripts/evaluate_all.sh

# Compare baseline vs looped
python scripts/compare_baseline.py \
  --baseline-checkpoint checkpoints/baseline/latest.pt \
  --looped-checkpoint checkpoints/looped/latest.pt

# Fine-tune on real video (after adding clips to data/flights/)
./scripts/launch_training.sh configs/aerojepa_finetune.yaml

# Resume interrupted fine-tune
./scripts/launch_training.sh configs/aerojepa_finetune.yaml \
    --resume checkpoints/real_finetune/latest.pt

# Capture Tello footage (hardware + djitellopy required)
python scripts/capture_tello.py --out-dir data/flights

# Demo
python app.py --checkpoint checkpoints/world_model/latest.pt

# Tests
pytest -q
```

---

## Suggested next steps (you decide)

Pick based on what matters most right now — **visibility**, **credibility**, or **capability**.

### A. Make it public (visibility)

- [ ] `git add` + initial commit (repo is initialized but largely untracked)
- [ ] Push to GitHub `JMangold0352/aerojepa`
- [ ] Pin best figures in README (gallery section like looped-jepa)
- [ ] Optional: Hugging Face Space from `app.py`

**Why:** Gets it in front of recruiters, defense contractors, and collaborators fast. Synthetic numbers are already impressive.

### B. Real drone data (credibility)

- [ ] Fly Tello, capture 10–20 clips → `data/flights/`
- [ ] Train `configs/aerojepa_real.yaml` (50 epochs)
- [ ] Fine-tune world_model from synthetic checkpoint (needs `--init-checkpoint` — quick code add)
- [ ] Report sim-to-real gap honestly in `REPORT.md`

**Why:** This is what separates “cool repo” from “relevant to autonomy.”

### C. Science depth (publication / ablations)

- [x] Review `results/ablations/summary.json` (20-epoch suite done)
- [ ] Re-run ablations at 100 epochs for paper-quality table
- [ ] Add comparison figure (bar chart) to README from ablation JSON
- [ ] Write short arXiv-style note using `docs/TECHNICAL_REPORT.md` as base

**Why:** Strengthens the recurrent-predictor story with controlled variants.

### D. Planning & sim (capability demo)

- [ ] Install PyFlyt, wire `sim/planner.py` into a hover task
- [ ] Show latent rollout → planned action → sim step (even a short GIF)
- [ ] Optional: obstacle-avoidance cost head instead of placeholder smoothness cost

**Why:** “World model that plans” is the defense pitch in one demo.

### E. Polish & maintenance

- [x] `--init-checkpoint` on `scripts/train.py`
- [x] `--resume` on `scripts/train.py` (Prompt 7)
- [x] `scripts/launch_training.sh` for reliable long runs
- [ ] Kill stale `monitor_training.sh` process if still running (harmless but noisy)
- [ ] `.gitignore` already excludes checkpoints — decide whether to release weights via Git LFS or instructions-only

---

## Honest limitations (say these out loud)

- Trained on **procedural synthetic** clips, not real aerial video.
- **No closed-loop flight** — research code, not a flight controller.
- **Action conditioning** did not separate on synthetic data.
- **No git commit/push yet** — work is local.
- Ablation at 20 epochs is a **quick signal** (done; all ~0.99), not publication-final.

---

## Where to read more

| Doc | Contents |
| --- | --- |
| [README.md](README.md) | Main project page, results table, quickstart |
| [REPORT.md](REPORT.md) | Experiment log |
| [REPRODUCTION.md](REPRODUCTION.md) | Reproduce from scratch |
| [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) | Architecture deep dive |
| [PROJECT_SCOPE.md](PROJECT_SCOPE.md) | Living scope + training playbook (updated post–Prompt 7) |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Five-phase plan |
| [docs/CURSOR_KICKOFF_PROMPTS.md](docs/CURSOR_KICKOFF_PROMPTS.md) | Cursor prompts for Phase 2+ |
| [model_cards/](model_cards/) | Per-model documentation |
| [results/README.md](results/README.md) | Metrics index |

---

## My read on priority

If the goal is **defense-space notice**: **A (public GitHub + README gallery) + B (one real Tello session)** in that order. Synthetic numbers and the looped-vs-baseline win are already a story; real footage makes it believable.

If the goal is **research publication**: **C** first, then B.

If the goal is **demo that wows in a meeting**: **D** with the Gradio app on a laptop (**A**’s demo path).

---

*This file is a snapshot. Update it when Phase 2 real-data runs land or when you choose a direction.*
