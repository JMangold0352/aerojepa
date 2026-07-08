# AeroProber: A Physics-Inspired Prober for Frozen Latent World Models

Technical note v1.0 -- July 2026. Prepared for PhD collaborator review.

## 1. Summary

AeroProber is a lightweight prober that converts latent rollouts from a frozen
AeroJEPA world model into physically meaningful metric states (position,
velocity, Euler attitude, angular velocity) through a differentiable kinematic
integrator. The encoder and (looped) predictor stay frozen; only the prober
(~5k parameters) is trained. This document compares the approach to SkyJEPA
(Rao et al., arXiv:2606.23444), presents ablation results, and lists open
questions for refinement.

## 2. Relation to SkyJEPA

SkyJEPA introduces a prober head that decodes latent representations into
metric state for aerial robotics. AeroProber takes the same core idea and:

- **Applies it to a looped latent world model.** SkyJEPA probed a single-pass
  encoder; we probe a recurrent (looped) predictor whose latent rollout is
  refined over up to 3 iterations. The looped-vs-regular comparison (Section 5)
  tests whether adaptive compute in the world model improves metric
  groundability -- a question SkyJEPA does not address.
- **Uses Euler angles, not SO(3)/quaternion, for v1.** This matches the
  AeroJEPA telemetry convention and keeps the integrator simple. We flag the
  SO(3) upgrade as the most important refinement for the PhD to advise on
  (Section 6).
- **Trains on PyFlyt simulator ground truth.** Real Parrot Wilds footage lacks
  absolute position, so we train on PyFlyt (full 6-DoF state) and evaluate
  quantitatively on Wilds for velocity/attitude/altitude only.

## 3. Architecture

```
PyFlyt clip (frames, actions, metric_state)
        |
   [frozen encoder] -- context frames -> context latents (192-d)
        |
   [frozen predictor] -- (regular max_loops=1 OR looped max_loops=3)
        |                 -> per-frame target latents (192-d), mean-pooled
        |
   [prober MLP, ~5k params] -- latents + actions -> residual accelerations
        |                       (3 linear + 3 angular)
   [kinematic integrator] -- nominal physics (action-derived) + residual
        |                       -> metric trajectory (pos, vel, att, ang_vel)
        |
   MSE vs ground-truth trajectory (wrapped-angle attitude loss)
```

### 3.1 Kinematic integrator (Euler)

State: `s = (pos, vel, euler_att, ang_vel)` where attitude is `(yaw, pitch, roll)`
in degrees, wrapped to `(-180, 180]`.

Nominal acceleration from the AeroJEPA 6-DoF action
`(dx, dy, d_altitude, d_yaw, d_pitch, d_roll)`:

```
a_lin_nom = (action[:, :3] - vel) / dt + (0, 0, g)
a_ang_nom = (action[:, 3:] - ang_vel) / dt
```

The action's body-velocity channels are treated as a first-order hold on
velocity (the action *is* the velocity), so the implied acceleration is the
velocity change over `dt`. Gravity `g = -9.81 m/s^2` acts on world-z so that a
zero-action drone falls -- the prober must learn the residual to counteract it.

Prober residual: `res_lin, res_ang = prober(latents, actions)` (each 3-D).

Euler step:
```
vel'  = vel + (a_lin_nom + res_lin) * dt
pos'  = pos + vel' * dt
ang_vel' = ang_vel + (a_ang_nom + res_ang) * dt
att'  = wrap_degrees(att + ang_vel' * dt)
```

### 3.2 Loss

Multi-horizon MSE over the predicted trajectory against ground truth, with a
**wrapped-angle** error for attitude (so a 359->1 prediction is a 2-degree
error, not 358):

```
loss = mean( [pos_err, vel_err, wrap(att_err), ang_vel_err]^2 )
```

## 4. Ablation: structured prober vs plain MLP vs naive

Three decoder arms on the **regular** (single-pass) predictor, 5 seeds, paired
test clips (64 training clips, 15 epochs):

| Arm | Params | Position RMSE (m) | Attitude RMSE (deg) | Velocity RMSE (m/s) |
| --- | --- | --- | --- | --- |
| naive (linear) | 2,316 | 0.058 +/- 0.003 | 2.92 +/- 0.06 | 0.246 +/- 0.004 |
| plain MLP | 5,076 | 0.147 +/- 0.014 | 2.86 +/- 0.06 | 0.282 +/- 0.011 |
| structured (ours) | 4,926 | 0.138 +/- 0.022 | **1.74 +/- 0.002** | 0.505 +/- 0.106 |

**Pre-registered success criterion:** structured position RMSE < plain position
RMSE with non-overlapping std bands across seeds. **Result: NOT met on position**
(bands overlap: 0.138 vs 0.147). **Met on attitude** (1.74 vs 2.86, non-overlapping).

### Interpretation

- **Attitude is the real win.** The structured prober reduces attitude RMSE by
  39% vs plain MLP with extremely tight variance (std 0.002 deg). The kinematic
  integrator's attitude structure (wrapped Euler integration + angular-velocity
  dynamics) is clearly carrying signal the plain MLP cannot match.
- **Position is a wash between structured and plain.** Both are far behind the
  naive linear projection, which is surprisingly strong (0.058 m). This suggests
  position is nearly linearly decodable from the frozen latent -- the integrator
  structure does not help (and may slightly hurt) when the latent already
  contains position information.
- **Velocity RMSE is worse for structured.** The integrator's first-order
  velocity hold (action = velocity target) may be too crude; the prober's
  residual has to fight the nominal model. This is a candidate for PhD input
  (Section 6, Q2).

See `results/prober_regular_ablation/summary.json` and
`figures/error_vs_horizon.png`.

## 5. Looped vs regular predictor

The same structured prober is run with the looped predictor's latent rollout
(`max_loops=3`, exit gate on) and compared against the regular baseline
(`max_loops=1`). Same seeds, same test clips. This tests whether the looped
predictor's refined latents yield more accurate metric trajectories -- the
project's headline research question.

Results (5 seeds, 64 clips, 15 epochs):

| Arm | Metric | Regular | Looped | Looped better? |
| --- | --- | --- | --- | --- |
| structured | position RMSE (m) | 0.1382 | 0.1477 | no |
| structured | attitude RMSE (deg) | 1.7368 | 1.7367 | yes (marginal) |
| structured | velocity RMSE (m/s) | 0.5051 | 0.4606 | yes |
| naive | position RMSE (m) | 0.0575 | 0.0529 | yes |
| plain | attitude RMSE (deg) | 2.8635 | 2.8599 | yes (marginal) |

### Headline finding

**The looped predictor does NOT meaningfully improve metric-trajectory accuracy
for the structured prober.** Position RMSE is slightly worse (0.138 -> 0.148);
attitude RMSE is essentially identical (1.7368 -> 1.7367 deg, a 0.0001 deg
difference far within noise); velocity RMSE improves modestly (0.505 -> 0.461).

This is a **negative result** for the project's central hypothesis: the looped
predictor's adaptive compute, which demonstrably refines latent predictions
(verified in `test_looped_checkpoint_max_loops_1` -- max_loops=1 vs 3 produce
different latents), does not translate into better metric groundability through
the prober. The frozen encoder's latent already contains the metric information
the prober can extract; the looped predictor's refinement helps latent
prediction but not metric decoding.

This is itself a publishable finding -- it bounds the value of adaptive compute
in latent world models for the metric-decoding regime. See `results/regular_vs_looped/`.

## 6. Open questions for the PhD

1. **Attitude representation.** Euler angles have gimbal-lock issues for
   aggressive maneuvers. Should v2 move to SO(3) exponential map or quaternion
   integration? What is the right trade-off between integrator complexity and
   physical fidelity for this data regime?

2. **Drag modeling.** The nominal model currently has no aerodynamic drag --
   the prober's residual absorbs everything the nominal model misses. Should we
   add a simple quadratic drag term to the nominal model so the prober's
   residual is smaller and more interpretable?

3. **Loss formulation.** We use uniform MSE over all horizons. Should we
   weight short horizons more heavily (the planner cares most about near-term)?
   Or add a divergence penalty for long-horizon stability?

4. **Ablation suggestions.** What other ablations would strengthen the
   contribution? Candidates: (a) prober depth/width, (b) integrator dt, (c)
   gravity on/off, (d) action-conditioned vs unconditioned predictor.

5. **Physical plausibility check.** Do the integrated trajectories look
   physically plausible on real Wilds footage? Are there systematic biases
   (e.g. always overshooting altitude) that hint at missing physics?

## 7. Reproducibility

- Branch: `feature/aeroprober`
- Configs: `research/prober/configs/`
- Run: `python research/prober/scripts/run_ablations.py --config research/prober/configs/prober_synth.yaml`
- Tests: `pytest research/prober/tests/`
- Checkpoints: frozen AeroJEPA `.pt` files (gitignored; see `checkpoints/`)

## 8. Limitations and future work

- **Position on real data.** Parrot Wilds has no GPS, so position x/y is
  dead-reckoned and qualitative only. A GPS-preserving capture pipeline is the
  key follow-up for real-data position RMSE.
- **Sim-to-real gap.** The prober is trained on PyFlyt (sim) and evaluated on
  Wilds (real). Domain randomization in PyFlyt could narrow this gap.
- **Single drone, single camera.** No multi-agent or multi-camera reasoning.
- **Short horizons.** 4-frame prediction (the encoder's `num_frames=8` minus
  4 context). Longer horizons need a recurrent prober or latent autoregression.
