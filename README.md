<div align="center">

# AeroJEPA: Recurrent Video World Models for Embodied UAV Autonomy

**A compact video-JEPA world model for drones — extending the [looped-jepa](https://github.com/JMangold0352/looped-jepa) recurrent predictor from single images into space and time.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-3776ab?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c?logo=pytorch&logoColor=white)](pyproject.toml)
[![Gradio demo](https://img.shields.io/badge/demo-Gradio-f97316?logo=gradio&logoColor=white)](app.py)
[![Params ~3–5M](https://img.shields.io/badge/params-~3--5M-54A24B)](configs/aerojepa_synth_base.yaml)
[![Best real cosine 0.974](https://img.shields.io/badge/real%20footage-latent%200.974-54A24B)](results/real_finetune_fast_eval.json)
[![License: research](https://img.shields.io/badge/license-research%20%2F%20educational-lightgrey)](#license)

*Self-supervised video encoders · latent future prediction · adaptive-depth recurrence · sim-to-real on aerial footage · latent-space planning*

**Representation (unconditioned):** `checkpoints/real_finetune_fast/latest.pt` — protocol-B real latent cosine **0.974** · rollout @ h=4 **0.984** · gap **+0.019** ([`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)).

**Closed-loop stack:** `checkpoints/action_conditioned_wilds/latest.pt` + `checkpoints/action_residual_wilds/best.pt` — gradient planner, `latent_smooth=0.05` (wind / turn / recover / hover cleared on 3-seed table).

[Quickstart](#installation--quickstart) ·
[Results](#key-results) ·
[Sim-to-real](#sim-to-real-transfer) ·
[Gallery](#visual-gallery) ·
[Demo](#gradio-demo) ·
[Model cards](model_cards/) ·
[Report](REPORT.md) ·
[Eval protocol](docs/EVAL_PROTOCOL.md) ·
[Reproduce](REPRODUCTION.md)

Parent project: [**looped-jepa**](https://github.com/JMangold0352/looped-jepa)

</div>

---

## Overview

AeroJEPA is a compact video-JEPA world model for drones. It extends the
[looped-jepa](https://github.com/JMangold0352/looped-jepa) recurrent predictor
from single images into space and time: predict future latents from short
egocentric clips, optionally conditioned on 6-DoF motion.

The repo includes synthetic training (no downloads), Wilds / Tello real-data
paths, eval and visualization scripts, a Gradio demo, and a PyFlyt latent
planner. Models stay around **3–5M** parameters.

---

## Key results

Official metric: **latent cosine similarity** between the predictor and EMA teacher on held-out clips (higher is better). JSON metrics: [`results/`](results/).

### Best representation — real aerial footage (`real_finetune_fast`)

Fine-tuned from synthetic pretrain on **15 Parrot clips** (The Wilds Drones; **128×128 on disk**, **64×64 model input**, 10 epochs, MPS). Checkpoint: [`checkpoints/real_finetune_fast/latest.pt`](checkpoints/real_finetune_fast/latest.pt).

| Metric | Value | Notes |
| --- | ---: | --- |
| **Latent cosine (real, protocol B)** | **0.974** | `scripts/evaluate_real.py` on `data/flights_128` |
| **Latent cosine (synthetic)** | **0.994** | Same script, synthetic branch |
| **Real rollout @ h=4** | **0.961** | Flat over horizon |
| **Sim-to-real gap** | **+0.019** | synth − real cosine |
| **Planner coherence** | **1.000** | Open-loop hover / waypoint on this checkpoint |

Full eval: [`results/real_finetune_fast_eval.json`](results/real_finetune_fast_eval.json) · Figures: [`visualizations/figures/real_finetune_fast/`](visualizations/figures/real_finetune_fast/) · Protocol: [`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md)

### Closed-loop stack (action-conditioned Wilds + residual)

Use this for PyFlyt demos, not the unconditioned representation checkpoint.

| Piece | Path |
| --- | --- |
| World model | `checkpoints/action_conditioned_wilds/latest.pt` |
| Residual | `checkpoints/action_residual_wilds/best.pt` |
| Planner | `--planner gradient --latent-smooth 0.05` |

Protocol-B real cosine for the WM alone is **0.957** (weaker than `real_finetune_fast`; keeps 6-DoF conditioning). Multi-seed table: [`visualizations/closed_loop/full_stack_compare_wilds.json`](visualizations/closed_loop/full_stack_compare_wilds.json). Card: [action Wilds](model_cards/aerojepa_action_wilds.md).

### Synthetic pretrain (100 epochs, procedural benchmark)

| Model | Objective | Latent cosine | Rollout @ h=4 | Expected loops | Card |
| --- | --- | ---: | ---: | ---: | --- |
| **baseline** | masked | 0.954 | 0.917 | — | [base](model_cards/aerojepa_base.md) |
| **looped** | masked | **0.961** (+0.007) | 0.925 | 1.50 | [base](model_cards/aerojepa_base.md) |
| **world_model** | future | **0.981** | 0.973 | 1.75 | [world model](model_cards/aerojepa_world_model.md) |
| action_conditioned | future + 6-DoF | 0.980 | 0.975 | 1.75 | [world model](model_cards/aerojepa_world_model.md) |

**Headline gains (synthetic)**

- **Recurrence helps on video:** looped beats feed-forward baseline by **+0.7 pp** latent cosine.
- **Future objective learns:** rollout stays **flat ~0.97** over a 4-frame horizon.
- **Refinement pays off:** per-loop cosine **0.87 → 0.96 → 0.98**; exit gate averages **1.75** steps.
- **Action conditioning** is roughly tied with the unconditioned world model on synthetic data; real telemetry is the better test.

### Ablation suite (100 epochs, publication)

| Variant | Latent cosine | Rollout @ h=4 | Per-loop (1→last) |
| --- | ---: | ---: | --- |
| baseline | 0.953 | 0.917 | — |
| loops_2 | 0.960 | 0.925 | 0.941 → 0.960 |
| loops_3 | 0.963 | 0.928 | 0.841 → 0.962 |
| **world_model** | **0.981** | **0.973** | 0.869 → 0.981 |

Details: [`results/ablations/summary.json`](results/ablations/summary.json) · [REPORT.md](REPORT.md)

---

## Sim-to-real transfer

AeroJEPA closes the synthetic → real loop in three stages. Each stage reuses the same `(frames, actions)` training interface — only the data source changes.

```mermaid
flowchart LR
  synth["Synthetic pretrain\n(world_model)"] --> wilds["Wilds fine-tune\n(15 Parrot clips)"]
  wilds --> tello["Tello fine-tune\n(personal footage)"]
  wilds --> eval["evaluate_real.py\nsim-to-real gap"]
  tello --> report["tello_compare_report.md\nvs Wilds-only baseline"]
```

| Stage | Data | Config | Output |
| --- | --- | --- | --- |
| **1. Synthetic** | Procedural drone clips (no download) | `aerojepa_world_model.yaml` | `checkpoints/world_model/latest.pt` |
| **2. Wilds** | [The Wilds Drones](https://huggingface.co/datasets/imageomics/thewilds_drones) (Parrot MP4 + telemetry) | `aerojepa_finetune_fast.yaml` | `checkpoints/real_finetune_fast/latest.pt` |
| **3. Tello** | Self-captured sessions (hover, forward, turn, altitude) | `aerojepa_finetune_tello.yaml` | `checkpoints/real_finetune_tello/latest.pt` |

**Measured transfer (Wilds stage, protocol B):** real latent cosine **0.974**, synthetic **0.994**, gap **+0.019** (`docs/EVAL_PROTOCOL.md`).

```bash
# Report sim-to-real gap for any checkpoint
python scripts/evaluate_real.py \
    --checkpoint checkpoints/real_finetune_fast/latest.pt \
    --data-dir data/flights_128

# After Tello fine-tune: compare vs Wilds-only baseline
python scripts/tello_compare_report.py
# → results/tello_transfer_report.md

# Transfer curve (data volume vs gap / rollout):
python scripts/run_transfer_curve.py --device mps
# → results/transfer_curve/transfer_curve.png
```

One-command Tello pipeline: [`scripts/tello_workflow.sh`](scripts/tello_workflow.sh) · Data layout: [`data/README.md`](data/README.md)

---

## Visual gallery

All figures at **300 DPI** (PNG) or animated GIF. Regenerate from checkpoints, then refresh [`docs/gallery/`](docs/gallery/) for README embeds — see [`docs/gallery/README.md`](docs/gallery/README.md).

### Real-data world model (`real_finetune_fast`)

<table>
<tr>
<td align="center" width="50%">

**Rollout on real Parrot footage**

Flat cosine over a 4-frame horizon after Wilds fine-tune.

<img src="docs/gallery/real_rollout.png" width="420"/>

</td>
<td align="center" width="50%">

**Per-loop refinement (real data)**

Coarse → refined latent prediction: **0.83 → 0.96 → 0.99**.

<img src="docs/gallery/real_per_loop_cosine.png" width="420"/>

</td>
</tr>
<tr>
<td align="center" width="50%">

**Latent planner — hover**

Imagined rollout on the real-data checkpoint (coherence **1.000**).

<img src="docs/gallery/planner_hover_real.gif" width="420"/>

</td>
<td align="center" width="50%">

**Latent planner — waypoint**

Kinematic cost search in latent space (coherence **1.000**).

<img src="docs/gallery/planner_waypoint_real.gif" width="420"/>

</td>
</tr>
</table>

### Synthetic pretrain & ablations

<table>
<tr>
<td align="center" width="50%">

**World-model rollout (synthetic)**

Healthy forward model signature — no horizon cliff.

<img src="docs/gallery/world_model_rollout.png" width="420"/>

</td>
<td align="center" width="50%">

**Per-loop refinement (looped)**

Each extra loop improves latent prediction.

<img src="docs/gallery/looped_per_loop_cosine.png" width="420"/>

</td>
</tr>
<tr>
<td align="center" width="50%">

**Predictor attention across time**

Self-attention with frame boundaries.

<img src="docs/gallery/looped_attention.png" width="420"/>

</td>
<td align="center" width="50%">

**Latent planner (synthetic checkpoint)**

Observed context → imagined rollout → plan.

<img src="docs/gallery/planner_waypoint.gif" width="420"/>

</td>
</tr>
<tr>
<td align="center" colspan="2">

**Ablation comparison** — per-loop gain + rollout curves

<img src="docs/gallery/ablation_per_loop.png" width="420"/>
&nbsp;&nbsp;
<img src="docs/gallery/ablation_rollout.png" width="420"/>

</td>
</tr>
</table>

Full figure sets: [`visualizations/figures/`](visualizations/figures/) · Planner outputs: [`visualizations/planner/`](visualizations/planner/)

```bash
python scripts/visualize.py --checkpoint checkpoints/real_finetune_fast/latest.pt \
  --out-dir visualizations/figures/real_finetune_fast
python scripts/run_planner_demo.py --checkpoint checkpoints/real_finetune_fast/latest.pt --task waypoint
python scripts/run_planner_demo.py --checkpoint checkpoints/real_finetune_fast/latest.pt --task hover
```

---

## Installation & quickstart

```bash
git clone https://github.com/JMangold0352/aerojepa.git && cd aerojepa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # or: pip install -e .

python scripts/verify_install.py
pytest -q
```

No GPU required. Auto-selects **MPS** (Apple Silicon), CUDA, or CPU.

**Load encoder in Python**

```python
import torch
from aerojepa.eval import load_model

model, cfg = load_model("checkpoints/real_finetune_fast/latest.pt", torch.device("cpu"))
clip = torch.rand(1, cfg["data"]["num_frames"], 3, cfg["data"]["img_size"], cfg["data"]["img_size"])
features = model.encoder.forward_all_patches(clip)   # (1, T*S, D)
```

### Workflow A — Synthetic (zero downloads)

Train the full model ladder on procedural drone clips:

```bash
python scripts/train.py --config configs/aerojepa_baseline.yaml
python scripts/train.py --config configs/aerojepa_looped.yaml
python scripts/train.py --config configs/aerojepa_world_model.yaml
python scripts/evaluate.py --checkpoint checkpoints/world_model/latest.pt
python scripts/compare_baseline.py \
  --baseline-checkpoint checkpoints/baseline/latest.pt \
  --looped-checkpoint  checkpoints/looped/latest.pt
```

### Workflow B — Real footage (Wilds Parrot)

Ingest public aerial data, preprocess, fine-tune from synthetic pretrain:

```bash
# Convert HuggingFace Wilds clips → data/flights/
python scripts/convert_wilds.py --input-dir data/raw/thewilds --output-dir data/flights

# Standardize to 128×128 on disk / 15 fps (model still trains at img_size=64)
python scripts/preprocess_real.py --input-dir data/flights \
  --output-dir data/flights_128 --target-fps 15 --max-seconds 60 --square --resize 128

# Optional: fast frame-index selection via Rust (OpenCV still decodes/encodes)
#   curl https://sh.rustup.rs -sSf | sh && source "$HOME/.cargo/env"
#   uv pip install maturin && cd native/aerojepa-preprocess && maturin develop --release
#   python scripts/preprocess_real.py ... --backend auto   # uses Rust when installed
# See docs/NATIVE_PREPROCESS.md

# Fine-tune (~4 min on MPS for 10 epochs)
python scripts/train.py --config configs/aerojepa_finetune_fast.yaml --device mps

# Sim-to-real gap
python scripts/evaluate_real.py --checkpoint checkpoints/real_finetune_fast/latest.pt \
  --data-dir data/flights_128
```

Long runs: [`scripts/launch_training.sh`](scripts/launch_training.sh) · Resume: `--resume checkpoints/.../latest.pt`

### Workflow C — Personal Tello footage

Record-only capture (never commands flight). One command runs preflight → four tagged maneuvers → preprocess → fine-tune → transfer report:

```bash
pip install djitellopy opencv-python

./scripts/tello_workflow.sh preflight          # Wi-Fi, battery, deps
./scripts/tello_workflow.sh all --duration 30  # full pipeline

# Or step by step:
./scripts/tello_workflow.sh sessions --duration 30   # hover, forward, turn, altitude
./scripts/tello_workflow.sh preprocess               # → data/flights_tello_128/
./scripts/tello_workflow.sh train                    # from real_finetune_fast
./scripts/tello_workflow.sh report                   # vs Wilds-only baseline
```

Safety: the capture script **records only** — you fly manually. See [`data/README.md`](data/README.md).

All CLIs: [`scripts/README.md`](scripts/README.md) · Configs inherit from [`configs/aerojepa_synth_base.yaml`](configs/aerojepa_synth_base.yaml).

---

## Gradio demo

Interactive world-model demo: generate a synthetic flight, choose context frames and refinement loops, watch prediction quality respond. **Run Latent Planner** tab imagines action plans and renders the outcome.

```bash
pip install gradio
python app.py --checkpoint checkpoints/world_model/latest.pt
# → http://127.0.0.1:7860
```

| | |
| --- | --- |
| **Local** | [`app.py`](app.py) · [`demo/README.md`](demo/README.md) |
| **Planner CLI** | `python scripts/run_planner_demo.py --checkpoint checkpoints/real_finetune_fast/latest.pt --task waypoint` |
| **Closed-loop (default stack)** | `python scripts/run_closed_loop_demo.py --checkpoint checkpoints/action_conditioned_wilds/latest.pt --residual-checkpoint checkpoints/action_residual_wilds/best.pt --planner gradient --latent-smooth 0.05 --task hover` |
| **Closed-loop waypoint** | same stack, `--task waypoint --goal 0.6 0 0` |
| **Closed-loop recover** | same stack, `--task recover` |
| **Demo reel** | `python scripts/stitch_closed_loop_demo.py --include-random` → `docs/gallery/closed_loop_demo_reel.gif` |

---

## Why the small model

Design targets that show up throughout the stack:

- Forward prediction in latent space (anticipate, don’t just classify)
- Synthetic → Wilds → Tello transfer, with a measured gap at each step
- Latent planning before execution (research demos in PyFlyt)
- Self-supervised pretrain; fine-tune from a small clip set
- Exit gate spends compute on hard moments only
- ~3–5M params so experiments stay cheap on a laptop

This is research code, not a deployed flight system.

---

## How it works

```mermaid
flowchart LR
  clip["Drone clip + telemetry"] --> enc["Context encoder"]
  clip --> teacher["EMA teacher (full clip)"]
  enc --> pred["Looped predictor + exit gate"]
  action["6-DoF motion (optional)"] --> pred
  pred --> loss["Latent Smooth-L1"]
  teacher --> loss
```

Two objectives, one network — only the mask collator changes:

- **`masked`** — scattered hidden patches (representation learning)
- **`future`** — predict upcoming frame latents (forward world model)

Deep dive: [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md)

---

## Model cards

Per-model documentation: architecture, training recipe, metrics, limitations, and load-and-run snippets.

| Card | Summary |
| --- | --- |
| [**aerojepa_base**](model_cards/aerojepa_base.md) | Baseline + looped masked objective, recurrence gains |
| [**aerojepa_world_model**](model_cards/aerojepa_world_model.md) | Future-frame objective, rollout, action conditioning |
| [**aerojepa_real_finetune**](model_cards/aerojepa_real_finetune.md) | Unconditioned Wilds fine-tune (`real_finetune_fast`) |
| [**aerojepa_action_wilds**](model_cards/aerojepa_action_wilds.md) | Action-conditioned Wilds + residual closed-loop stack |
| [**Index**](model_cards/README.md) | All cards |

---

## Limitations

| Limitation | Status |
| --- | --- |
| Closed-loop is a research demo | PyFlyt + heuristic map + small residual; not a flight controller |
| Action-conditioned Wilds vs unconditioned | ~0.957 vs ~0.974 real cosine (protocol B); action model kept for planning |
| Tello fine-tune | Workflow ready; no personal footage on disk yet |
| Checkpoints | Local artifacts; not in git ([REPRODUCTION.md](REPRODUCTION.md)) |
| Onboard / ROS / motor-command hardware | Not in scope for now |

**Near term**

1. Physics collaboration on AeroProber — start from [`research/prober/note.md`](research/prober/note.md)
2. Tello sessions when you can capture (`./scripts/tello_workflow.sh`)
3. Keep the default closed-loop stack on **v1** Wilds action + residual (v2 regressed)

Experiment log: [`REPORT.md`](REPORT.md)

---

## Repository layout

```
aerojepa/
├── src/aerojepa/       models · data · train · eval · viz · sim
├── configs/            synthetic · real · finetune · tello
├── scripts/            train · evaluate · tello_workflow · visualize · planner
├── docs/gallery/       README-embedded figures (tracked)
├── app.py · demo/      Gradio demo + latent planner tab
├── model_cards/        per-model documentation
├── results/            JSON metrics (tracked); checkpoints/ runs/ data/ gitignored
└── docs/               technical notes · gallery
```

---

## Citation

```bibtex
@misc{mangold2026aerojepa,
  title        = {AeroJEPA: Recurrent Video World Models for Embodied UAV Autonomy},
  author       = {John Mangold},
  year         = {2026},
  howpublished = {\url{https://github.com/JMangold0352/aerojepa}},
  note         = {Video JEPA with weight-shared recurrent predictor, extending looped-jepa}
}
```

**Acknowledgments** — [I-JEPA](https://arxiv.org/abs/2301.08243) · [Vision Transformer](https://arxiv.org/abs/2010.11929) · [looped-jepa](https://github.com/JMangold0352/looped-jepa) · [The Wilds Drones](https://huggingface.co/datasets/imageomics/thewilds_drones)

---

## Documentation index

| Doc | Contents |
| --- | --- |
| [docs/NATIVE_PREPROCESS.md](docs/NATIVE_PREPROCESS.md) | Optional Rust frame-index helper (deferred beyond Phase A) |
| [docs/EVAL_PROTOCOL.md](docs/EVAL_PROTOCOL.md) | How published metrics are produced |
| [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) | Architecture and design notes |
| [data/README.md](data/README.md) | Real footage layout, Tello, preprocess |
| [REPRODUCTION.md](REPRODUCTION.md) | Reproduce from scratch |
| [REPORT.md](REPORT.md) | Experiment log |
| [results/README.md](results/README.md) | Metrics index |
| [scripts/README.md](scripts/README.md) · [visualizations/README.md](visualizations/README.md) · [demo/README.md](demo/README.md) | Component guides |

---

## License

Research and educational use. See config headers for experiment lineage.
