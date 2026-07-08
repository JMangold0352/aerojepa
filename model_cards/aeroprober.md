# Model card: AeroProber (physics-inspired prober)

## Overview

AeroProber is a lightweight prober head that converts latent rollouts from a
**frozen** AeroJEPA world model into physically meaningful metric states
(position, velocity, Euler attitude, angular velocity) through a differentiable
kinematic integrator. The encoder and (looped) predictor stay frozen; only the
prober (~5k parameters) is trained.

- **Parent model:** [AeroJEPA-WorldModel](aerojepa_world_model.md) (frozen).
- **Code:** [`research/prober/`](../research/prober/)
- **Technical note:** [`research/prober/note.md`](../research/prober/note.md)

## Architecture

| Component | Setting |
| --- | --- |
| Frozen encoder | ViT, 192-d, 6 blocks (from parent checkpoint) |
| Frozen predictor | regular (single-pass) or looped (max_loops=3, exit gate) |
| Prober (psi) | MLP, ~5k params (hidden=24, 2 layers) |
| Integrator | first-order Euler, Euler-angle attitude, wrapped to (-180, 180] |
| Prober output | residual accelerations (3 linear + 3 angular) |
| Loss | multi-horizon MSE with wrapped-angle attitude error |

## Intended use

- **Metric rollout:** decode frozen latents into position/velocity/attitude
  trajectories for planning and visualization.
- **Ablation research:** compare structured physics prober vs plain MLP head
  vs naive linear projection (see `research/prober/scripts/run_ablations.py`).
- **Looped-vs-regular comparison:** test whether the looped predictor's refined
  latents yield more accurate metric trajectories.

## Training

- **Data:** PyFlyt quadrotor simulator (full 6-DoF metric state ground truth).
- **Frozen:** encoder + predictor + EMA teacher (zero trainable params).
- **Trained:** prober head only.
- **Recipe:** AdamW, lr=1e-3, 30 epochs, grad clip 1.0. See
  [`research/prober/configs/prober_synth.yaml`](../research/prober/configs/prober_synth.yaml).

## Evaluation

- **Synthetic (PyFlyt):** position RMSE, attitude RMSE (deg), velocity RMSE,
  per-horizon error curves. See `research/prober/results/`.
- **Real (Parrot Wilds):** velocity/attitude/altitude RMSE (quantitative);
  position x/y qualitative only (no GPS in Wilds logs). See
  `research/prober/scripts/eval_real.py`.

## Limitations

- Euler angles (not SO(3)/quaternion) -- gimbal lock possible for aggressive
  maneuvers. Flagged for v2.
- Real-data position is dead-reckoned (no GPS). Flagged as future work.
- Trained on sim (PyFlyt), evaluated on real (Wilds) -- sim-to-real gap exists.

## Relation to SkyJEPA

Extends SkyJEPA's prober concept to a looped latent world model and evaluates
whether adaptive compute (the looped predictor) improves metric groundability.
See `research/prober/note.md` for the full comparison.
