# Gating + body residual experiment

**Status:** scripted; run `python research/prober/scripts/run_gating_exp.py`.  
**JEPA:** frozen. SkyJEPA (Rao et al., arXiv:2606.23444) uses a world-frame
\(\Delta\dot v\) residual; this body-vs-world invariance test is **not** in their paper.

## Variants

| ID | Name | Residual |
| --- | --- | --- |
| A | Ungated world | \(a_w^\mathrm{res}=\mathrm{MLP}(s,u)\) (current AeroProber default) |
| B | Gated body | \(\dot v=(T/m)Re_3-g+R\,a_b^\mathrm{res}(s,u)\) |
| C | Partial \(r_z\) | Only body-\(z\) residual rotated through \(R\); xy zeroed in body |

## Frame test

Rotate the inertial frame by a known \(R_\mathrm{extra}\in\mathrm{SO}(3)\):

- Body residual must be **invariant**.
- World residual must **rotate** as \(R_\mathrm{extra}a_w\).

Implemented in `aerojepa_research.prober.gating.frame_invariance_test`.

## Prediction

Gated body should win at **1-5 s** with a \(t^2\) signature
(\(\sim\tfrac12 g\theta t^2\)), not necessarily at the headline **0.1 s**
(4 frames, \(\Delta t=0.025\)).

## How to run

```bash
# Invariance only (no PyFlyt):
python research/prober/scripts/run_gating_exp.py --skip-train

# Short train/eval (PyFlyt outside sandbox):
python research/prober/scripts/run_gating_exp.py \
  --epochs 5 --num-train 64 --num-val 32 --device cpu
# → research/prober/results/gating_exp/gating_results.json
```

Metrics: geodesic attitude RMSE, position RMSE vs horizon, error vs \(t\) and \(t^2\).

## Results (short run, 3 epochs, 32 train / 16 val, 0.1 s horizon)

See `results/gating_exp/gating_results.json`. At headline **0.1 s** (as predicted,
gating need not win yet):

| variant | pos RMSE (m) | geodesic att (°) |
| --- | ---: | ---: |
| A ungated world | 0.0051 | 2.69 |
| B gated body | 0.0055 | 2.69 |
| C partial \(r_z\) | 0.0057 | 2.69 |

Frame invariance: **pass** (body invariant; world must rotate).
Longer-horizon \(t^2\) comparison is still open.
