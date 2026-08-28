# Correctness notes (AeroProber / closed-loop plant)

**Date:** 27 Aug 2026  
**Scope:** leak-free probing, control semantics, frames, attitude metric, Wilds data.

This is a public checklist of what the code does. Line numbers refer to the
tree as of this date and may drift; search the cited symbols if needed.

---

## Plant

| Quantity | Value | Source |
| --- | --- | --- |
| Controls \(c\) | \((v_p,v_q,v_r,T)\) body-rate setpoints (rad/s) + normalized thrust | `research/prober/.../integrator.py` (`ControlIntegrator`) |
| Not | 4 rotor forces (SkyJEPA), PWM, or AeroJEPA 6-DoF pose deltas | - |
| Attitude storage | Euler yaw-pitch-roll degrees, wrap \((-180,180]\) | `MetricState` |
| Mass / hover | \(m=1\,\mathrm{kg}\), \(T_\mathrm{hover}=0.39\) | `ControlIntegrator.__init__` |
| \(\Delta t\) | \(0.025\,\mathrm{s}\) (40 Hz) | configs / ctor |
| Headline horizon | 4 predict frames = **0.1 s** | clip `num_frames=8`, context 4 |
| Linear residual | world-frame | `ControlIntegrator.step` |
| Angular residual | body-frame, typically ×0.25 | `Prober.forward` |

Closed-loop map \(x^+ = f_\mathrm{cl}(x, u_\mathrm{cmd})\), not Newton-Euler in rotor forces.
Legacy `KinematicIntegrator` uses state-derived 6-DoF actions - **leaky**; not for
headline v3 numbers.

---

## V1 - Leak / freeze / splits

| Check | Result | Citation |
| --- | --- | --- |
| Encoder frozen | Pass - `requires_grad=False`, `eval()`, extract under `@torch.no_grad()` | `rollout.py` (`FrozenRolloutExtractor`) |
| Prober controls | Pass (v3) - raw `(vp,vq,vr,T)`, not `states_to_actions` | `data_pyflyt.py`, `prober.py` |
| Future frames | Pass - context tokens only into the encoder | `FrozenRolloutExtractor.extract` |
| Train/val | Pass - disjoint PyFlyt clip seeds (`seed` vs `seed+9973`) | `build_pyflyt_dataloaders` |

`hover` / `kick` / `turn` dataset modes are state-dependent PD (leaky if enabled).
Default recipes keep those fractions at 0; `PyFlytClipsDataset` warns otherwise.

---

## V2 - Attitude metric

| Check | Result |
| --- | --- |
| Training loss | Wrapped Euler MSE (`loss.py`) |
| Reporting | Dual-report: legacy wrapped Euler RMSE **and** geodesic `Log` degrees |

Use `attitude_rmse_geodesic_deg` / `geodesic_attitude_error_deg` in
`metrics.py` for new tables. Older published tables used wrapped Euler only.

---

## V3 - What \(u\) is

Rate + thrust `(vp, vq, vr, T)`. No mixer or command delay in-repo; lag is
absorbed into the residual. Not PWM and not four rotor forces.

---

## V4 - Body vs world

| Quantity | Frame |
| --- | --- |
| `pos`, `vel` (metric GT) | world (PyFlyt body `lin_vel` rotated by \(R\)) |
| `ang_vel` (metric GT) | body `(p,q,r)` deg/s (PyFlyt rad/s → deg) |
| Linear residual | world |
| Angular residual / rate setpoints | body |

Implemented in `_obs_to_metric_state` (`data_pyflyt.py`).

AeroProber position RMSE tables (~0.006 m at 0.1 s) were
trained **before** this GT unit/frame correction. Re-run ablations before
treating post-fix numbers as comparable to those tables.

---

## V5 - Attitude update

Default plant: body rates → Euler chart rates (`body_rates_to_euler_rates_deg`)
then wrap. Not \(R\leftarrow R\exp([\omega]_\times\Delta t)\). Chart is singular near
pitch ±90°. Structure-preserving alternatives live in `so3_integrators.py`
(Exp / RK4 / LGVI bake-off under `research/prober/results/integrator_bakeoff/`).

---

## V6 - Wilds / metadata

- `data/flights_128/` is often a **transcoded** 128 px preprocess; model input remains 64×64.
- Converter can symlink original Anafi MP4s and read mission **JSON** telemetry; it does not parse binary `.vmeta`.
- Wilds `ang_vel` from attitude deltas (YPR deg/s) differs from PyFlyt body `(p,q,r)`.

---

## Comparisons

- Short-horizon probe error is not vehicle tracking accuracy.
- AeroProber ~0.006 m (0.1 s sim) is not comparable to SkyJEPA outdoor open-loop RMSE.
- Looped vs regular predictor on the structured prober: **tie**.
- No energy-conservation claims for a forced, draggy quadrotor.
- Action-conditioned Wilds does **not** yet pass true/zero/shuffle counterfactuals
  (`results/action_counterfactual.json`); counterfactual tests haven't passed.
