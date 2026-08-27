# AeroProber: A Physics-Inspired Prober for Frozen Latent World Models

Technical note v2.0 -- July 2026.

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

**Citation.** Rao, Zhang, Balestriero, LeCun, and Loianno, *SkyJEPA: Learning
Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors*,
arXiv:2606.23444, 2026. https://arxiv.org/abs/2606.23444

```bibtex
@article{rao2026skyjepa,
  title   = {SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors},
  author  = {Rao, Pratyaksh and Zhang, Wancong and Balestriero, Randall and LeCun, Yann and Loianno, Giuseppe},
  journal = {arXiv preprint arXiv:2606.23444},
  year    = {2026},
  doi     = {10.48550/arXiv.2606.23444},
  url     = {https://arxiv.org/abs/2606.23444}
}
```

SkyJEPA is a *state*-history JEPA (pose/twist + rotor forces) with a
physics-inspired prober and outdoor MPPI. AeroJEPA / AeroProber ask whether
*video* latents contain rigid-body state — SkyJEPA’s own conclusion names RGB as
future work. **Do not** compare our short-horizon sim pos RMSE (~0.006 m) to
SkyJEPA outdoor RMSE (~1.43 m): different horizon, \(\Delta t\), speed, and relative
vs outdoor setting.

SkyJEPA introduces a prober head that decodes latent representations into
metric state for aerial robotics. AeroProber takes the same *family* of idea and:

- **Applies it to a video latent world model** (optionally looped). The
  looped-vs-regular comparison (Section 5) tests whether adaptive compute
  improves metric groundability — currently a **tie**.
- **Uses a physics-structured residual integrator.** The prober predicts
  residual accelerations on top of a nominal thrust/torque model, rather than
  decoding state directly.
- **Uses Euler angles for v1** (matching AeroJEPA telemetry). SO(3) upgrade
  is an open question (Section 9).
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
# Thrust is normalized; hover_thrust cancels |g| when level (mass cancels):
a_lin_nom = R @ [0, 0, (T/hover_thrust)*|g|] + [0, 0, g]
a_ang_nom = (rad2deg(c[:3]) - ang_vel_pqr) / dt    # body rates, first-order hold
# Chart step uses body→Euler rate map (not Exp); see so3_integrators bake-off.
```

Prober residual: `res_lin, res_ang = prober(latents, controls)`.

Euler step: integrate world linear accel; convert body \(\omega\) to Euler rates;
wrap attitude.

Parameters: `dt = 0.025 s` (PyFlyt 40 Hz), `g = -9.81`, `mass = 1.0 kg`,
`hover_thrust = 0.39`.

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
  the prober's residual adds noise on attitude. Open question (Section 8).

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

**Eval protocol (v4, leak-free):** structured prober trained on PyFlyt with frozen
`real_finetune_fast` (`configs/prober_real_finetune.yaml`); evaluated with
`eval_real.py`. Contiguous frame windows, `dt = 1/fps`, GT body-frame velocity
rotated to world before scoring, angular residuals gated (`0.25`), normalized
thrust scaled so hover ≈ cancels gravity. Controls at eval:

- **zeros + weaker nominal** (`gravity=0`): motors unknown; do not force free-fall
  in the nominal model (Section 8).
- **hover prior** (optional): constant exogenous `T=0.39`, rates zero — not
  GT velocity / pose-delta actions.

**Results v3 (legacy protocol)** — for provenance only:

| Metric | Wilds v3 | PyFlyt sim (v3) | Gap |
| --- | --- | --- | --- |
| velocity RMSE (m/s) | 1.29 ± 0.62 | 0.075 | ~17× |
| attitude RMSE (deg) | 33.6 ± 29.4 | 2.28 | ~15× |
| altitude RMSE (m) | 5.69 ± 6.97 | — | — |

**Results v4** (`results/real_data_v4/`, 15 clips, zero-control + weaker nominal):

| Metric | Wilds v4 | vs v3 | ≥50% cut? |
| --- | --- | --- | --- |
| velocity RMSE (m/s) | **0.135** | **−89.5%** | **MET** (target ≤0.645) |
| attitude RMSE (deg) | **0.923** | **−97.3%** | **MET** (target ≤16.8°) |
| altitude RMSE (m) | **0.057** | −99% | — |

Hover-prior eval (`results/real_data_v4_hover/`): vel **0.177**, att **0.923**.

Prober still **4878** params. No GT pose-delta actions; controls remain exogenous
zeros or a constant thrust prior.

### Interpretation

Most of the v3 gap was **protocol / physics mismatch**, not latent failure:

1. Uniform whole-clip sampling with `dt=0.025` while frames were seconds apart
   blew up attitude integration.
2. Body-frame Parrot velocity compared to world-frame predictions.
3. Normalized PyFlyt thrust treated as Newtons → nominal free-fall dominated
   zero-control rollouts.

After fixing those (leak-free), residual Wilds error is small on short horizons.
Remaining gap vs sim (vel 0.135 vs 0.075) is domain shift plus missing motor
commands — the main open problem for longer horizons and SO(3).

## 8. Scientific evals (Aug 2026)

### Action counterfactuals

`scripts/eval_action_counterfactual.py` on `action_conditioned_wilds`.
**Verdict: fail** — true ≈ zero ≈ shuffle on latent cosine/L1 (~0.994). Do **not**
claim a causal action-conditioned world model yet. See
`results/action_counterfactual.json`.

### Compounding

`scripts/eval_compounding.py`: open-loop / teacher-forced latent L1; CR grows
with horizon. Physics-only overlay at 0.1 s: short-horizon pos RMSE relative to
predict-window \(t=0\), \(\Delta t=0.025\). Do not compare to SkyJEPA outdoor
tracking RMSE.

### Hard PyFlyt (10 seeds, v1 stack)

`scripts/run_hard_pyflyt_suite.py` → `visualizations/closed_loop/stress_suite.json`.
Wind ≤4 m/s and recover/hover kicks still ~100%; **L-turn scale ×1.25 collapses
to 0%** (cliff). Prefer this figure over any 100%×3-seed table.

### Gating / integrator

- Frame invariance: pass (`gating_exp.md`).
- At 0.1 s, ungated vs gated variants not yet separated (as expected).
- Integrator bake-off: Euclidean Euler leaves SO(3); Exp / RK4 / LGVI preserve
  \(\|R^\top R-I\|_F\).

## 9. Open questions (physics)

See also [`docs/CORRECTNESS.md`](../../docs/CORRECTNESS.md). Headline table uses
**0.1 s** horizon (\(\Delta t=0.025\), 4 frames). Attitude in published v3 tables is
wrapped Euler RMSE; prefer geodesic going forward (`metrics.attitude_rmse_geodesic_deg`).

1. **Attitude residual gating.** Underactuation: \(a = R(e_3 T/m)-ge_3+R f\).
   Ungated world \(\Delta\dot v\) can memorize texture. Gated body residual must
   commit to \(R\) (or \(r_z\)) first. Prediction: gated wins at 1–5 s with a
   \(t^2\) signature, not at 0.1 s. Angular residual already scaled by 0.25.

2. **SO(3) parameterization.** Bake-off on the *same* residual net: Euler chart /
   quat / 6D / \(\mathfrak{so}(3)\)+Exp. Primary metric: geodesic error and
   \(\|R^\top R - I\|_F\), including pitch \(\approx 90^\circ\).

3. **What \(u\) is.** Ours is rate+thrust \(c=(v_p,v_q,v_r,T)\), not SkyJEPA’s
   four rotor forces — a closed-loop map. Identifiability battery: GT forces /
   motor lag / rate+thrust / shuffled delay; when are \(m,J\) recoverable?

4. **Body vs world.** Video and \(\omega\) are body; gravity and \(p\) are
   inertial. Current linear residual is **world-frame**. Test: rotate the
   inertial frame — body residual must be invariant; world residual must rotate.

5. **Horizon vs integrator.** Freeze residual net; swap Euler-on-\(\mathbb{R}^9\) /
   Exp / RK4-on-\(\mathfrak{so}(3)\) / LGVI. Horizons 0.2 / 1 / 5 / 10 s including
   zero-thrust coasts. Claim constraint preservation, never energy conservation.

## 10. Reproducibility

- Branch: `feature/aeroprober`
- Leak-free ablation: `python research/prober/scripts/run_ablations.py --config research/prober/configs/prober_synth.yaml --seeds 0 1 2 3 4 --num-train 256 --epochs 30 --output-dir research/prober/results/prober_regular_ablation_full_v3`
- Real eval (v4): `python research/prober/scripts/eval_real.py --prober research/prober/results/prober_real_finetune/best.pt --checkpoint checkpoints/real_finetune_fast/latest.pt --data-dir data/flights_with_state --controls zeros`
- Optional hover prior: add `--controls hover`
- Tests: `pytest research/prober/tests/`
- **Do not use** `*_full_v2` results as headline numbers (information leak).
- **Do not use** Wilds `real_data_v3` as the current headline (legacy protocol).

## 11. Limitations

- Euler angles, single quadrotor, short horizons (4 predict frames).
- Wilds position x/y qualitative only.
- Sim-trained prober, real-evaluated; domain gap is the main open problem.
- Invalid v2 results archived for provenance only.
