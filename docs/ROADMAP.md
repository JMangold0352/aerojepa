# AeroJEPA Roadmap

A pragmatic, five-phase path from the current architecture-and-benchmark release
to sim-based predictive planning. Effort estimates are rough and assume a single
researcher with a laptop-class machine plus a DJI Tello.

| Phase | Focus | Key deliverables | Status | Est. effort |
| --- | --- | --- | --- | --- |
| **1** | Architecture & data pipeline | Video JEPA model, synthetic generator, training loop, eval + viz, demo | **Done** | 2-3 weeks |
| **2** | Real data & training | Ingest real UAV footage (Tello / MotionScape / AeroVerse), full training runs, ablations | **In progress** | 3-4 weeks |
| **3** | Interpretability & demo polish | Attention-over-time studies, latent-trajectory analysis, hosted demo | After Phase 2 | 1-2 weeks |
| **4** | Simulation & planning | Close the loop in PyFlyt / gym-pybullet-drones; latent-space planning; task metrics | After Phase 3 | 2 weeks |
| **5** | Documentation & release | Model cards with real numbers, paper-ready figures, public release | Final | 1-2 weeks |

## Phase 1 - Architecture & data pipeline (done)

The current release: space-time tokenization, the recurrent predictor with an
exit gate and sandwich RMSNorm, masked and future objectives, optional 6-DoF
action conditioning, a synthetic drone-clip generator, the full evaluation
harness, the figure suite, and an interactive demo. Everything trains end-to-end
from a clean checkout with no downloads.

## Phase 2 - Real data & training

- Wire real footage through [`data/video_dataset.py`](../src/aerojepa/data/video_dataset.py)
  and capture Tello clips via [`data/tello.py`](../src/aerojepa/data/tello.py).
- Add loaders for public UAV corpora (MotionScape, AeroVerse-style benchmarks).
- Run longer training and the ablation suite at scale; log honest numbers.
- Deliverable: the first real-data entries in [REPORT.md](../REPORT.md) and the
  model cards.

## Phase 3 - Interpretability & demo polish

- Attention-across-time and per-loop studies on real footage.
- Latent-trajectory visualizations for real flights.
- Host the Gradio demo (e.g. Hugging Face Spaces via `app.py`).

## Phase 4 - Simulation & planning

- Close the loop in a simulator through
  [`sim/simulators.py`](../src/aerojepa/sim/simulators.py).
- Replace the placeholder cost in
  [`sim/planner.py`](../src/aerojepa/sim/planner.py) with a task cost (e.g. a
  learned collision-risk head) and evaluate latent-space planning.
- Deliverable: a task-success metric that shows the world model helps.

## Phase 5 - Documentation & release

- Model cards with measured numbers, paper-ready figures, and a public release.

## Explicitly out of scope (for the foreseeable future)

Fully autonomous flight without human supervision, multi-drone swarms, heavy RL,
deployment on severely constrained hardware, and outdoor BVLOS / regulatory
work. These are important but well beyond the research goals of this project.
