# AeroJEPA Technical Report

*Status: kickoff / Phase 1. This document describes the architecture and design
rationale of the initial release. Experimental results are produced by the
evaluation harness and logged in [REPORT.md](../REPORT.md).*

## 1. Motivation

Perception for autonomy is usually framed as recognition -- label the frame. But
an autonomous drone needs something recognition does not give it: the ability to
**anticipate**. If the system can predict the near future of its own sensory
stream, it can plan around obstacles, react to motion, and do so with far less
labeled data than a supervised detector would need.

JEPA (Joint-Embedding Predictive Architecture) is a natural fit. Rather than
reconstructing pixels -- which wastes capacity on textures and noise that do not
matter for control -- JEPA predicts the **latent** representation of hidden
content. The training signal comes from an exponential-moving-average (EMA)
teacher, so no labels and no contrastive negatives are required.

The parent project, [looped-jepa](https://github.com/JMangold0352/looped-jepa),
established that a **recurrent** predictor -- one small block stack re-applied
several times, with a learned exit gate -- adds effective depth and per-sample
adaptivity at **zero additional parameters**, and that a paired ("sandwich")
RMSNorm is what makes that recurrence stable at small scale. AeroJEPA extends
that predictor from single images into video, and from gap-filling into forward
prediction.

## 2. Architecture

### 2.1 Space-time tokenization

A clip `(B, T, C, H, W)` is converted into `T' x S` tokens:

- **Frame tokenizer** (default): a 2D patch convolution is applied to every
  frame independently, giving `S = (H/patch)^2` spatial tokens per frame and
  `T' = T` temporal steps. Simple and easy to reason about.
- **Tubelet tokenizer** (optional, `tokenizer: tubelet`): a 3D convolution over
  `(tubelet_size, patch, patch)` bakes short-range motion into each token and
  reduces the temporal token count to `T' = T / tubelet_size`.

Positions use a **factorized space-time encoding**: a spatial table `(S, D)` and
a temporal table `(T', D)` are added, so token index `t * S + s` carries both
"where" and "when". Factorization keeps the positional parameter count at
`(S + T') * D` instead of `S * T' * D`.

### 2.2 Encoder and EMA teacher

The **context encoder** is a standard ViT that operates on an arbitrary *subset*
of tokens (gathered by index) -- exactly what masked prediction needs. The
**target encoder** is an EMA copy that encodes the full clip with gradients
stopped; it provides the prediction targets. Only the context encoder is kept for
downstream use.

### 2.3 Recurrent predictor

The predictor is a narrow ViT structured as a single reusable `BlockStack`. It
places learned **mask tokens** at target positions, concatenates them with the
projected context tokens, and runs the stack. The `LoopedVideoPredictor` wraps it
and re-applies the stack up to `max_loops` times; a per-loop **exit gate**
(sigmoid on the pooled hidden state) yields adaptive depth. An **exit-entropy
regularizer** prevents the gate from collapsing to a constant.

The world-model recipe (`world_model: true`) uses RMSNorm + SwiGLU + sandwich
norm -- the configuration that was most stable under recurrence in the parent
project.

### 2.4 Action conditioning

For model-based control, the predictor optionally consumes the drone's 6-DoF
motion. Per-frame action vectors are projected to the predictor width and added
to every token belonging to that frame (via the same gather used for positions).
This turns "fill in the blanks" into "given this motion, what comes next?".

## 3. Training objectives

The network is objective-agnostic; the **mask collator** decides the task:

- **Masked** (`SpatioTemporalMaskCollator`): targets are space-time blocks
  scattered across random frames. General representation learning (V-JEPA style).
- **Future** (`FutureFrameMaskCollator`): context is every token of the first
  `num_context_frames`; targets are every token of the remaining frames. A
  forward world model.

Loss is smooth-L1 in latent space (robust to occasional large targets), plus the
exit-entropy term when the gate is active. Optimization is AdamW with a linear
warmup and cosine decay; the EMA momentum follows a cosine ramp toward (but
capped below) 1.0.

## 4. The synthetic benchmark

To make the project reproducible with zero downloads, `data/synthetic.py`
procedurally renders drone clips: a static, textured overhead "world" (multi-
octave value noise, earthy tint, a few colored obstacle disks) viewed through a
moving 6-DoF virtual camera. Egomotion (translation, altitude/zoom, yaw) is a
smooth random walk with edge reflection; the per-frame pose deltas are the
telemetry/action signal. The physics are simple but contain the structure a world
model must capture: coherent motion, parallax-like scaling, and looming
obstacles. `data/video_dataset.py` is a drop-in replacement for real footage.

## 5. Evaluation

- **Latent prediction** (`eval/latent_pred.py`): cosine and smooth-L1 between
  predicted and teacher latents.
- **Rollout** (`eval/rollout.py`): cosine as a function of prediction horizon;
  the shape of this curve characterizes the world model.
- **Loop analysis** (`eval/loop_metrics.py`): per-loop cosine (is recurrence
  refining?), exit distribution, and expected loops (compute actually spent).

## 6. Limitations and honesty

- The synthetic benchmark is a *proxy*. It validates the pipeline and the
  scientific question, but real aerial video is the true test (roadmap Phase 2).
- Latent prediction quality is not the same as task success; downstream planning
  metrics (Phase 4) are needed to claim autonomy value.
- The planner in `sim/planner.py` is a readable reference (random shooting with a
  placeholder cost), not a flight controller.

See [REPORT.md](../REPORT.md) for the running experiment log, including negative
results, and [ROADMAP.md](ROADMAP.md) for what comes next.
