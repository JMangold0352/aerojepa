# Model card: AeroProber (physics-inspired prober)

## Overview

AeroProber is a lightweight prober head that converts latent rollouts from a
**frozen** AeroJEPA world model into physically meaningful metric states
(position, velocity, Euler attitude, angular velocity) through a differentiable
**control integrator**. The encoder and predictor stay frozen; only the prober
(~5k parameters) is trained.

- **Parent model:** [AeroJEPA-WorldModel](aerojepa_world_model.md) (frozen).
- **Code:** [`research/prober/`](../research/prober/)
- **Technical note:** [`research/prober/note.md`](../research/prober/note.md)

## Architecture

| Component | Setting |
| --- | --- |
| Frozen encoder | ViT, 192-d, 6 blocks (from parent checkpoint) |
| Frozen predictor | regular (single-pass) or looped (`max_loops=3`) |
| Prober (ψ) | MLP, ~5k params (`hidden=24`, 2 layers) |
| Prober input | latent (192-d) + **raw control** `(vp, vq, vr, T)` - 4-d, leak-free |
| Integrator | `ControlIntegrator`: thrust/torque nominal model + residual accelerations |
| Prober output | residual linear + angular acceleration (3 + 3) |
| Loss | multi-horizon MSE with wrapped-angle attitude error |

## Intended use

- **Metric rollout:** decode frozen latents into position/velocity/attitude trajectories.
- **Ablation research:** structured physics prober vs plain MLP vs naive linear probe.
- **Looped-vs-regular:** test whether looped predictor latents improve metric decoding.

## Training

- **Data:** PyFlyt quadrotor simulator (full 6-DoF metric state + raw control commands).
- **Frozen:** encoder + predictor (zero trainable params).
- **Trained:** prober head only.
- **Recipe:** AdamW, lr=1e-3, 30 epochs, `dt=0.025` (40 Hz). See `configs/prober_synth.yaml`.

## Evaluation (leak-free v3, full scale)

Caption note: horizon = 4 predict frames, \(\Delta t = 0.025\,\mathrm{s}\) (40 Hz),
position relative to \(t=0\) in-sim. **Not** comparable to SkyJEPA outdoor RMSE.

**Synthetic (PyFlyt), regular predictor, 5 seeds:**

| arm | position RMSE | attitude RMSE | velocity RMSE |
| --- | --- | --- | --- |
| structured | **0.006 ± 0.000 m** | **2.28 ± 0.00 deg** | **0.075 m/s** |
| plain MLP | 0.152 ± 0.022 m | 2.84 ± 0.03 deg | 0.252 m/s |

Pre-registered success criterion (structured position < plain, non-overlapping): **met**.

These tables were trained **before** the PyFlyt GT unit/frame fix
(body vel→world, rates rad→deg). See [`docs/CORRECTNESS.md`](../docs/CORRECTNESS.md).

**Looped vs regular (structured):** tie - negative result on adaptive compute helping metric groundability.

**Real (Parrot Wilds), 15 clips** (`results/real_data_v3/`):

| Metric | Wilds | PyFlyt sim (v3) |
| --- | --- | --- |
| velocity RMSE | 1.29 ± 0.62 m/s | 0.075 m/s |
| attitude RMSE | 33.6 ± 29.4 deg | 2.28 deg |
| altitude RMSE | 5.69 ± 6.97 m | - |

Eval uses zero controls (no motor telemetry). Large sim-to-real gap remains.

## Limitations

- Euler angles (not SO(3)) - gimbal lock possible for aggressive maneuvers.
- Real-data position x/y is dead-reckoned (no GPS).
- Sim-to-real gap: prober trained on PyFlyt, evaluated on Wilds.
- Wilds eval uses zero controls (motor commands unavailable); nominal physics is gravity-only at eval time.
- Earlier v2 results used state-derived actions and are **invalid** (information leak); use v3 only.

## Relation to SkyJEPA

[SkyJEPA](https://arxiv.org/abs/2606.23444) (Rao et al., 2026) is a state-history
JEPA with a physics-inspired prober and outdoor MPPI. AeroProber asks the same
*family* of question for **video** latents. Key addition: looped-vs-regular
metric groundability (currently a **tie**). Short-horizon sim RMSE here is not
comparable to SkyJEPA outdoor numbers. See `research/prober/note.md`.
