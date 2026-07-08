# AeroJEPA — Project Scope & Decision Brief

**Author:** John Mangold  
**Last updated:** July 3, 2026 (evening, post–Prompt 7)  
**Repo:** `~/Projects/aerojepa`

This is the living scope document: what the project is, what's done, what's next, and how to train seriously on a Mac Studio.

---

## Mission

Build a **recurrent video world model for UAV autonomy** — predict future structure in latent space, plan in that space, and close the sim-to-real gap with real drone footage. Extends [looped-jepa](https://github.com/JMangold0352/looped-jepa).

---

## Phase status

| Phase | Scope | Status |
| --- | --- | --- |
| **1** | Architecture + synthetic benchmark | **Complete** — 4 models @ 100 epochs, full eval/viz/demo |
| **2** | Real data + transfer | **Pipeline complete** — capture, preprocess, fine-tune; awaiting footage |
| **3** | Large corpus (60h+) | **Planned** — curriculum + subsampling when logs arrive |
| **4** | Closed-loop sim | **Scaffolded** — latent planner + PyFlyt hooks |
| **5** | Public release / paper | **Not started** — README gallery ready |

---

## What's done (July 3, 2026)

### Core ML
- ViT encoder + looped video predictor + exit gate (~3–5M params)
- Four trained synthetic checkpoints (best: `world_model`, latent cosine **0.981**)
- Masked + future objectives; action-conditioned variant
- 20-epoch ablation suite + publication figure pipeline

### Real-data pipeline (Prompt 6)
- `scripts/capture_tello.py` — record-only Tello capture + safety
- `scripts/preprocess_real.py` — probe, standardize, align telemetry
- `VideoClipDataset` — uniform/sliding windows, video-level val split
- `--init-checkpoint` fine-tune path + `aerojepa_finetune.yaml`

### Infrastructure (Prompt 7)
- **`--resume`** on `scripts/train.py` — restores model, optimizer, scheduler, epoch
- **`scripts/launch_training.sh`** — reliable long runs via Terminal.app + caffeinate
- **37 tests** passing (includes resume + preprocess)
- Config inheritance via `_base_` YAML (all train scripts use `load_config`)

---

## Training commands (serious runs)

```bash
cd ~/Projects/aerojepa && source .venv/bin/activate

# Quick sanity
python scripts/verify_install.py && pytest -q

# Fine-tune on real footage (recommended first real run)
./scripts/launch_training.sh configs/aerojepa_finetune.yaml

# Resume after interruption
./scripts/launch_training.sh configs/aerojepa_finetune.yaml \
    --resume checkpoints/real_finetune/latest.pt

# Monitor
tail -f logs/train_aerojepa_finetune_latest.log
```

**Important:** Use **Terminal.app** or `launch_training.sh`, not detached Cursor `nohup` (jobs die silently).

---

## Real-data workflow

```
capture (Tello)  →  preprocess  →  fine-tune  →  evaluate_real (sim-to-real gap)
```

```bash
python scripts/capture_tello.py --preflight
python scripts/capture_tello.py --duration 30 --fps 15 --name-prefix session1
python scripts/preprocess_real.py --probe --input-dir data/flights
python scripts/preprocess_real.py --input-dir data/flights --square
./scripts/launch_training.sh configs/aerojepa_finetune.yaml
python scripts/evaluate_real.py --checkpoint checkpoints/real_finetune/latest.pt
```

---

## 60-hour footage plan (few months)

When flight-control logs arrive:

1. **Never delete raw** — `data/raw/<flight_id>/` with video + logs + `metadata.json`
2. **Preprocess to** `data/flights/` with manifests (`train.txt`, `val.txt`)
3. **Train with curriculum** — subsample ~10k windows/epoch, not full 800k+ sliding windows
4. **Eval on fixed benchmark** — 10–20 held-out clips only, not full corpus each epoch
5. **Write `convert_control_logs.py`** once log schema is known

---

## Suggested next steps (ranked)

| Priority | Task | Why |
| ---: | --- | --- |
| 1 | Capture 10 Tello sessions | First real numbers |
| 2 | Fine-tune via `launch_training.sh` | Prove sim→real transfer |
| 3 | Sim-to-real transfer curve figure | Great README asset |
| 4 | PyFlyt closed-loop planner eval | Defense demo |
| 5 | 100-epoch ablation `--mode full` | Publication table |
| 6 | GitHub push + checkpoint release | Visibility |

---

## Elite upgrades (token-efficient)

- **Closed-loop eval** — planner + PyFlyt hover success rate
- **Transfer curve** — fine-tune on 1/5/20 clips, plot gap
- **Exit-gate calibration** — uncertainty vs error plot
- **Checkpoint release** — runnable without training

---

## Honest limitations

- Synthetic-trained until real fine-tune runs
- Not a flight controller — research code only
- Action conditioning untested on real telemetry
- No git push yet

---

## Key paths

| Path | Purpose |
| --- | --- |
| `checkpoints/world_model/latest.pt` | Best synthetic base for fine-tune |
| `configs/aerojepa_finetune.yaml` | Real-data fine-tune recipe |
| `data/README.md` | Footage layout + Tello safety |
| `docs/gallery/` | README figures (tracked) |
| `logs/train_*_latest.log` | Active training log symlink |

---

*Update this file when milestones land (first real fine-tune, 60h ingest, closed-loop demo).*
