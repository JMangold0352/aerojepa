# AeroJEPA — Project Status & Handoff

**Last updated:** 2026-07-03  
**Repo:** `~/Projects/aerojepa`  
**Parent project:** [looped-jepa](https://github.com/JMangold0352/looped-jepa) (recurrent I-JEPA on CIFAR-10)

This document summarizes everything built and measured so far, so you can decide what to do next without re-reading the whole chat.

---

## In one sentence

AeroJEPA extends your Looped-JEPA recurrent predictor into a **video world model for drones**: it predicts future scene structure in latent space (not pixels), trains end-to-end on a built-in synthetic flight generator, and now has **trained checkpoints + published numbers** on that synthetic benchmark.

---

## What we built (Phase 1 — complete)

### Core idea

- **JEPA-style training:** context encoder sees part of a clip; EMA teacher provides targets; predictor fills in missing/future latents.
- **Recurrent predictor:** weight-shared loops + learned exit gate (from looped-jepa), now over **space-time tokens**.
- **Two objectives:** `masked` (representation learning) and `future` (forward world model).
- **Optional 6-DoF action conditioning** for model-based control / planning.
- **Synthetic drone generator:** procedural 6-DoF camera over textured terrain + obstacles — **zero downloads** to train.

### Repository layout (high level)

```
aerojepa/
├── src/aerojepa/          # models, data, train, eval, viz, sim
├── configs/               # YAML recipes (_base_ inheritance)
├── scripts/               # train, evaluate, visualize, ablations, demo, …
├── checkpoints/           # trained weights (gitignored)
├── results/               # eval JSON + comparison
├── visualizations/figures/
├── model_cards/           # professional model docs
├── demo/ + app.py         # Gradio demo
├── docs/                  # technical report, roadmap, this file
└── tests/                 # 13 tests, all passing at last run
```

### Key configs

| Config | Purpose |
| --- | --- |
| `configs/aerojepa_synth_base.yaml` | Base recipe (everything inherits this) |
| `configs/aerojepa_baseline.yaml` | Feed-forward predictor, masked objective |
| `configs/aerojepa_looped.yaml` | Recurrent + exit gate + sandwich RMSNorm |
| `configs/aerojepa_world_model.yaml` | Future-frame objective, world-model predictor |
| `configs/aerojepa_action_conditioned.yaml` | World model + 6-DoF telemetry |
| `configs/smoke_test.yaml` | Tiny 1-epoch sanity check |
| `configs/aerojepa_real.yaml` | **Phase 2:** train on `data/flights/` video |

### Training stack

- Python 3.11, PyTorch 2.12, **MPS** on Apple Silicon
- ~**3.2M** params (baseline/looped), ~**5.1M** (world model with SwiGLU recipe)
- 100 epochs, 1024 synthetic train clips, 64×64×8 frames

---

## What we trained & measured

### Full training run (2026-07-03, synthetic data)

All four models trained **100 epochs** on MPS. Total wall time ~1.5 hours for baseline + remaining three.

| Model | Objective | Latent cosine (val) | Rollout @ 4 frames | Expected loops | Checkpoint |
| --- | --- | ---: | ---: | ---: | --- |
| **baseline** | masked | **0.954** | 0.917 | — | `checkpoints/baseline/latest.pt` |
| **looped** | masked | **0.961** | 0.925 | 1.50 | `checkpoints/looped/latest.pt` |
| **world_model** | future | **0.981** | 0.973 | 1.75 | `checkpoints/world_model/latest.pt` |
| **action_conditioned** | future + 6-DoF | **0.980** | 0.975 | 1.75 | `checkpoints/action_conditioned/latest.pt` |

**Head-to-head (baseline vs looped):** +**0.007** latent cosine (`results/comparison.json`).

### Main scientific takeaways (synthetic)

1. **Recurrence helps on video** when paired with sandwich RMSNorm (+0.7 pp vs feed-forward baseline).
2. **World-model rollout is flat** ~0.97 over 4-frame horizon — no cliff (good sign for planning).
3. **Per-loop refinement is real** on world model: cosine **0.87 → 0.96 → 0.98** over 3 loops; exit gate uses **~1.75** steps on average.
4. **Action conditioning** did not separate on synthetic data (0.980 vs 0.981) — real Tello telemetry is the honest test.

### Evaluation artifacts

- `results/baseline_eval.json`
- `results/looped_eval.json`
- `results/world_model_eval.json`
- `results/action_conditioned_eval.json`
- `results/comparison.json`
- `results/README.md` — table index

Re-run anytime:

```bash
./scripts/evaluate_all.sh
```

### Figures (from real checkpoints, not smoke)

- `visualizations/figures/looped/` — clip, rollout, per-loop cosine, exit dist, latent trajectory, attention
- `visualizations/figures/world_model/` — same suite for world model

Regenerate:

```bash
python scripts/visualize.py --checkpoint checkpoints/looped/latest.pt --out-dir visualizations/figures/looped
python scripts/visualize.py --checkpoint checkpoints/world_model/latest.pt --out-dir visualizations/figures/world_model
```

---

## Phase 2 kickoff (in progress)

### Done

- **README, REPORT.md, model cards** updated with measured numbers (no more "_to be measured_").
- **`configs/aerojepa_real.yaml`** — train on folder of real `.mp4` clips.
- **`data/README.md`** — expected layout (`flight_001.mp4` + optional `flight_001.csv` telemetry).
- **`scripts/capture_tello.py`** — Tello capture stub (needs `djitellopy`).
- **`scripts/evaluate_all.sh`** — evaluate all four checkpoints in one go.
- **`docs/ROADMAP.md`** — Phase 2 marked "in progress".

### Possibly still running

- **20-epoch ablation suite** (`scripts/run_ablations.py`) was started in background.
  - Log: `logs/ablations.log`
  - Output (when done): `results/ablations/summary.json`
  - Variants: baseline, loops_2, loops_3, world_model

Check status:

```bash
tail -f logs/ablations.log
cat results/ablations/summary.json   # when complete
```

### Not done yet (Phase 2)

- No real drone footage trained yet (`data/flights/` is empty unless you add clips).
- No `--init-checkpoint` / fine-tune flag (docs mention manual load or future follow-up).
- No GitHub push / public release (repo is local, git init done, files mostly untracked unless you committed).
- Gradio demo not smoke-tested with trained checkpoint in this session.
- Ablation `summary.md` human write-up (only JSON when script finishes).

---

## Operational notes (lessons from training day)

### What worked

- Training **inside a Cursor background shell** with `python scripts/train.py` ran to completion (100 epochs × 4 models).
- **MPS** at ~4 it/s per epoch (~8 s/epoch, ~13 min per 100-epoch run for baseline).

### What failed

- **`nohup` / detached shell jobs** died around batch 21–27 with no error in log — likely Cursor parent shell teardown.
- **Fix for overnight runs:** use **Terminal.app**, not Cursor shell:

  ```bash
  cd ~/Projects/aerojepa && ./scripts/launch_in_terminal.sh
  ```

- Old **training monitor** (`scripts/monitor_training.sh`) may still be sleeping in background from earlier; safe to ignore or kill manually if you see it.

---

## Tests & verification (last known good)

```bash
python scripts/verify_install.py
pytest -q                    # 13 passed
python scripts/train.py --config configs/smoke_test.yaml
```

---

## Documentation index

| Doc | What it is |
| --- | --- |
| [README.md](../README.md) | Main entry, results table, quickstart |
| [REPORT.md](../REPORT.md) | Experiment log |
| [REPRODUCTION.md](../REPRODUCTION.md) | Reproduce from scratch |
| [docs/TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | Architecture deep dive |
| [docs/ROADMAP.md](ROADMAP.md) | 5-phase plan |
| [docs/CURSOR_KICKOFF_PROMPTS.md](CURSOR_KICKOFF_PROMPTS.md) | Prompts for future Cursor sessions |
| [model_cards/](../model_cards/) | Per-model documentation |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | How to extend |

---

## Suggested next steps (you decide)

Pick what matters most for **defense visibility** vs **research depth**:

### A. Make it public & polished (low effort, high visibility)

1. `git add` + commit + push to GitHub
2. Run demo: `python app.py --checkpoint checkpoints/world_model/latest.pt`
3. Pin best figures in README gallery (link to `visualizations/figures/world_model/`)
4. Wait for / paste ablation summary into `results/ablations/summary.md`

### B. Real data (Phase 2 core — highest credibility)

1. Fly Tello (or use any drone video), save to `data/flights/`
2. `python scripts/train.py --config configs/aerojepa_real.yaml`
3. Compare synthetic-pretrained vs scratch (fine-tune hook TBD)
4. Update REPORT + model cards with **real** numbers (expect sim→real gap)

### C. Science depth

1. Finish ablations at **100 epochs** (not just 20): `python scripts/run_ablations.py --epochs 100`
2. Longer synthetic runs with harder generator settings (`num_obstacles`, `max_speed` in config)
3. Attention-over-time study on real footage (roadmap Phase 3)

### D. Planning / sim (Phase 4 — "gets noticed" for autonomy)

1. Wire `sim/planner.py` + PyFlyt (`sim/simulators.py`)
2. Replace placeholder `smoothness_cost` with task-specific cost
3. Closed-loop eval in sim

### E. Publication / outreach

1. Technical blog or short paper using `docs/TECHNICAL_REPORT.md` + figures
2. Hugging Face Spaces for `app.py`
3. Defense-relevant one-pager: latent world models, label efficiency, interpretable loops

---

## Quick command cheat sheet

```bash
# Environment
cd ~/Projects/aerojepa && source .venv/bin/activate

# Train (synthetic)
python scripts/train.py --config configs/aerojepa_world_model.yaml

# Train (real video in data/flights/)
python scripts/train.py --config configs/aerojepa_real.yaml

# Evaluate everything
./scripts/evaluate_all.sh

# Compare baseline vs looped
python scripts/compare_baseline.py \
  --baseline-checkpoint checkpoints/baseline/latest.pt \
  --looped-checkpoint checkpoints/looped/latest.pt

# Demo
python app.py --checkpoint checkpoints/world_model/latest.pt

# All-day training (use Terminal.app)
./scripts/launch_in_terminal.sh
```

---

## Open questions (from the original plan)

1. Does recurrence help **more** on the `future` objective than `masked`? — **Yes on synthetic** (world model shows stronger per-loop gain).
2. Does action conditioning help rollout? — **Not on synthetic**; needs real telemetry.
3. Does the exit gate spend more depth on hard clips? — **Not yet tested** systematically (needs per-clip difficulty analysis).

---

*When you pick a direction, tell the assistant which letter (A–E) or your own priority — the repo is ready to execute any of them.*
