# AeroProber: A Physics-Inspired Prober for Frozen Latent World Models

Technical note v2.0 -- July 2026. Prepared for PhD collaborator review.

## 1. Summary

AeroProber is a lightweight prober that converts latent rollouts from a frozen
AeroJEPA world model into physically meaningful metric states (position,
velocity, Euler attitude, angular velocity) through a differentiable **control
integrator**. The encoder and (looped) predictor stay frozen; only the prober
(~5k parameters) is trained.

This note presents **leak-free** ablation results (full scale, 5 seeds), documents
an information-leak bug we found and fixed during development, compares to SkyJEPA,
and lists open questions for refinement.

## 2. Relation to SkyJEPA

SkyJEPA introduces a prober head that decodes latent representations into
metric state for aerial robotics. AeroProber takes the same core idea and:

- **Applies it to a looped latent world model.** SkyJEPA probed a single-pass
  encoder; we probe a recurrent (looped) predictor. The looped-vs-regular
  comparison (Section 5) tests whether adaptive compute improves metric
  groundability -- a question SkyJEPA does not address.
- **Uses a physics-structured residual integrator.** The prober predicts
  residual accelerations on top of a nominal thrust/torque model, rather than
  decoding state directly.
- **Uses Euler angles for v1** (matching AeroJEPA telemetry). SO(3) upgrade
  flagged for PhD input (Section 7).
- **Trains on PyFlyt** (full 6-DoF state) and evaluates on Wilds for
  velocity/attitude/altitude.

## 3. Architecture (leak-free design)

```
PyFlyt clip (frames, control_actions, metric_state)
        |
   [frozen encoder] -- context frames -> context latents (192-d)
        |
   [frozen predictor] -- regular (max_loops=1) OR looped (max_loops=3)
        |                 -> per-frame target latents (192-d), mean-pooled
        |
   [prober MLP, ~5k params] -- latents + controls -> residual accelerations
        |                       (3 linear + 3 angular)
   [ControlIntegrator] -- nominal thrust/torque physics + residual
        |                  -> metric trajectory (pos, vel, att, ang_vel)
        |
   MSE vs ground-truth trajectory (wrapped-angle attitude loss)
```

### 3.1 Why raw control commands (not state-derived actions)

The prober's action input must be **exogenous** -- it cannot contain ground-truth
state, or the integrator copies the answer from the input and the latent is not
tested.

We use PyFlyt's raw control commands `(vp, vq, vr, T)` -- angular-rate setpoints
and collective thrust, sampled from a RNG. These drove the simulation but are not
derived from the metric state.

**Important:** An earlier design used AeroJEPA-style pose-delta actions
(`vgx, vgy, vgz` = velocity, attitude deltas) as the prober input. That leaked
ground-truth velocity and attitude into the integrator. Full-scale results from
that design (v2 overnight run) are **invalid** and kept only for provenance.

### 3.2 ControlIntegrator

State: `s = (pos, vel, euler_att, ang_vel)` -- attitude in degrees, wrapped.

Nominal physics from control `c = (vp, vq, vr, T)`:

```
R = rotation_matrix(yaw, pitch, roll)   # body -> world
a_lin_nom = R @ [0, 0, T/mass] + [0, 0, g]
a_ang_nom = (rad2deg(c[:3]) - ang_vel) / dt    # first-order hold on rate setpoints
```

Prober residual: `res_lin, res_ang = prober(latents, controls)`.

Euler step (same as before): integrate accelerations, wrap attitude.

Parameters: `dt = 0.025 s` (PyFlyt 40 Hz), `g = -9.81`, `mass = 1.0 kg`.

### 3.3 Loss

Multi-horizon MSE with wrapped-angle attitude error over the predicted trajectory.

## 4. Information leak (development note)

During full-scale evaluation of the convention-fixed design, we noticed structured
prober position RMSE was suspiciously low (~0.007 m). A zero-residual probe
(showing the nominal model alone, with no prober) achieved similar accuracy, and
`action[:, :3] == gt_velocity` was literally `True`.

**Root cause:** state-derived AeroJEPA actions were fed to an integrator that
treated them as velocity/attitude targets.

**Fix:** `ControlIntegrator` + raw `control_actions` from PyFlyt. Leak probe
after fix: `control != gt_vel`, zero-residual nominal model has substantial error
(pos ~0.013 m vs 0.18 m GT drift over 4 frames).

This episode is itself instructive: the structured prober only demonstrates value
when the nominal model is honest and the action input is exogenous.

## 5. Ablation: structured vs plain MLP vs naive (leak-free, full scale)

**Setup:** regular predictor, 5 seeds, 256 training clips, 30 epochs, 32 test
clips, paired comparison. Results: `results/prober_regular_ablation_full_v3/`.

| Arm | Params | Position RMSE (m) | Attitude RMSE (deg) | Velocity RMSE (m/s) |
| --- | --- | --- | --- | --- |
| naive (linear) | 2,316 | 0.039 ± 0.007 | 2.89 ± 0.03 | 0.237 ± 0.005 |
| plain MLP | ~4.9k | 0.152 ± 0.022 | 2.84 ± 0.03 | 0.252 ± 0.016 |
| **structured (ours)** | ~4.9k | **0.006 ± 0.000** | **2.28 ± 0.00** | **0.075 ± 0.003** |

**Pre-registered success criterion:** structured position RMSE < plain with
non-overlapping std bands. **Result: MET** (0.006 vs 0.152, non-overlapping).

### Interpretation

- **Structured prober wins decisively on position and velocity** (~25× better
  position than plain MLP). The control integrator's physics structure lets the
  prober learn small residuals instead of full state from the latent.
- **Structured also wins on attitude** vs plain MLP (2.28° vs 2.84°).
- **vs zero-residual nominal model** (fast-run check): structured improves
  position/velocity ~50% over nominal-only, but is **slightly worse on attitude**
  (2.29° vs 1.96° nominal). The angular-rate nominal model is already decent;
  the prober's residual adds noise on attitude. Open question for PhD (Section 7).

## 6. Looped vs regular predictor

Same structured prober, looped checkpoint (`max_loops=3`) vs regular (`max_loops=1`).
Results: `results/regular_vs_looped_full_v3/`.

| Metric | Regular | Looped | Looped better? |
| --- | --- | --- | --- |
| structured position RMSE (m) | 0.00618 | 0.00617 | marginally yes |
| structured attitude RMSE (deg) | 2.285 | 2.286 | no |
| structured velocity RMSE (m/s) | 0.075 | 0.076 | no |

### Headline finding

**The looped predictor does NOT meaningfully improve metric groundability** for
the structured prober at full scale under the leak-free design. This confirms the
negative result from earlier (pre-leak) runs and is a publishable bound on the
value of adaptive compute for metric decoding.

## 7. Real-data evaluation (Wilds)

Parrot ANAFI footage with extended state CSVs (`wilds_state.py`). 15 clips,
quantitative metrics: velocity, attitude, altitude. Position x/y is
dead-reckoned (no GPS) -- not a headline metric.

**Eval protocol:** structured prober trained on PyFlyt with frozen
`real_finetune_fast` checkpoint (`configs/prober_real_finetune.yaml`);
evaluated on Wilds with `eval_real.py`. **Controls are zero** at eval time
(Parrot logs lack motor commands) -- nominal integrator applies gravity only.

**Results** (`results/real_data_v3/real_data_metrics.json`, 15 clips):

| Metric | Wilds (real) | PyFlyt sim (v3) | Gap |
| --- | --- | --- | --- |
| velocity RMSE (m/s) | 1.29 ± 0.62 | 0.075 | ~17× |
| attitude RMSE (deg) | 33.6 ± 29.4 | 2.28 | ~15× |
| altitude RMSE (m) | 5.69 ± 6.97 | — | — |

High variance across clips (attitude std 29.4°). Best clip (`wilds_009`):
att 0.08°, vel 0.57 m/s. Worst clips (`wilds_006`, `wilds_008`): att >98°.

### Interpretation

The sim-to-real gap is large and expected: the prober is trained on PyFlyt with
known control inputs; at Wilds eval it sees zero controls and must infer dynamics
entirely from the latent. Attitude errors are especially severe, likely due to
(1) zero-control nominal model mismatch, (2) body vs world frame differences,
(3) domain shift between PyFlyt and Parrot footage.

**This is the primary research frontier:** domain randomization, pseudo-control
estimation from IMU, or fine-tuning on real data.

## 8. Open questions for the PhD

1. **Attitude residual gating.** Structured prober slightly underperforms the
   zero-residual nominal model on attitude in sim. Should angular residuals be
   gated, scaled down, or dropped when the nominal model is already accurate?

2. **SO(3) / quaternion integrator.** Euler angles work for gentle PyFlyt motion
   but have wrap and gimbal-lock issues. What is the right v2 attitude representation?

3. **Real-data controls.** Wilds lacks motor telemetry; we use zero controls at
   eval. Options: estimate pseudo-controls from IMU, fine-tune on real data with
   a weaker nominal model, or collect motor commands.

4. **Body vs world frame.** PyFlyt state is world-frame; Parrot telemetry uses
   body-frame velocities. A body→world rotation in the nominal model may narrow
   sim-to-real gap.

5. **Loss weighting and longer horizons.** Uniform MSE over 4 predict frames;
   planner may care more about near-term. Stress-test at 16–32 frames.

## 9. Reproducibility

- Branch: `feature/aeroprober`
- Leak-free ablation: `python research/prober/scripts/run_ablations.py --config research/prober/configs/prober_synth.yaml --seeds 0 1 2 3 4 --num-train 256 --epochs 30 --output-dir research/prober/results/prober_regular_ablation_full_v3`
- Real eval: `configs/prober_real_finetune.yaml` + `scripts/eval_real.py`
- Tests: `pytest research/prober/tests/`
- **Do not use** `*_full_v2` results as headline numbers (information leak).

## 10. Limitations

- Euler angles, single quadrotor, short horizons (4 predict frames).
- Wilds position x/y qualitative only.
- Sim-trained prober, real-evaluated; domain gap is the main research frontier.
- Invalid v2 results archived for provenance only.
