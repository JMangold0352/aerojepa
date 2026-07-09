# AeroProber

Physics-inspired prober that converts frozen AeroJEPA latent rollouts into
metric states (position, velocity, Euler attitude, angular velocity) through a
differentiable control integrator.

## Quick start

```bash
# From repo root. PyFlyt must run OUTSIDE the Cursor sandbox.

# 1. Train the leak-free structured prober (regular predictor, PyFlyt):
python research/prober/scripts/train_prober.py \
    --config research/prober/configs/prober_synth.yaml --arm structured

# 2. Headline ablation (5 seeds, naive / plain / structured):
python research/prober/scripts/run_ablations.py \
    --config research/prober/configs/prober_synth.yaml \
    --seeds 0 1 2 3 4 --num-train 256 --epochs 30 \
    --output-dir research/prober/results/prober_regular_ablation_full_v3

# 3. Looped vs regular comparison:
python research/prober/scripts/compare_regular_looped.py \
    --regular research/prober/results/prober_regular_ablation_full_v3 \
    --looped research/prober/results/prober_looped_ablation_full_v3 \
    --output-dir research/prober/results/regular_vs_looped_full_v3

# 4. Real-data eval (Wilds Parrot footage):
python research/prober/scripts/train_prober.py \
    --config research/prober/configs/prober_real_finetune.yaml --arm structured
python research/prober/scripts/eval_real.py \
    --prober research/prober/results/prober_real_finetune/best.pt \
    --checkpoint checkpoints/real_finetune_fast/latest.pt \
    --data-dir data/flights_with_state
```

## Layout

```
research/prober/
  src/aerojepa_research/prober/
    integrator.py     ControlIntegrator (leak-free) + KinematicIntegrator (legacy)
    prober.py         Prober MLP (~5k params) + PlainMLPHead ablation arm
    data_pyflyt.py    PyFlyt generator: frames, AeroJEPA actions, metric state, control_actions
    rollout.py        frozen-model rollout extraction (regular + looped)
    loss.py           structured / plain / naive loss fns (wrapped-angle att)
    metrics.py        position/velocity/attitude RMSE + per-horizon curves
    wilds_state.py    extended Parrot Wilds converter (preserves abs state)
  scripts/
    train_prober.py   config-driven trainer (--arm structured|plain|naive)
    run_ablations.py  headline experiment (multi-seed, paired, figures)
    eval_real.py      real-data eval on Wilds (velocity/attitude/altitude RMSE)
  configs/
    prober_synth.yaml          sim training (frozen baseline checkpoint)
    prober_looped.yaml         looped predictor arm
    prober_real_finetune.yaml  sim training with frozen real_finetune_fast (for Wilds eval)
  tests/
  note.md             technical note for PhD review
  results/            JSON metrics + figures
```

## Design (v1, leak-free)

- **Prober input:** frozen latent rollout + **raw control commands** `(vp, vq, vr, T)` from PyFlyt. These are exogenous (sampled from RNG), not derived from ground-truth state.
- **Integrator:** `ControlIntegrator` maps thrust + angular-rate setpoints to nominal accelerations; prober predicts residuals.
- **Attitude:** Euler angles (yaw, pitch, roll), wrapped to `(-180, 180]`.
- **Ground truth:** PyFlyt (full 6-DoF). Wilds for real-data velocity/attitude/altitude only.
- **Frozen:** encoder + predictor. Only the prober trains (~5k params).

## Headline results (leak-free full-scale v3)

5 seeds, 256 clips, 30 epochs, regular predictor (`results/prober_regular_ablation_full_v3/`):

| arm | position RMSE (m) | attitude RMSE (deg) | velocity RMSE (m/s) |
| --- | --- | --- | --- |
| naive | 0.039 ± 0.007 | 2.89 ± 0.03 | 0.237 |
| plain MLP | 0.152 ± 0.022 | 2.84 ± 0.03 | 0.252 |
| **structured** | **0.006 ± 0.000** | **2.28 ± 0.00** | **0.075** |

Success criterion (structured pos < plain, non-overlapping bands): **MET**.

Looped vs regular (structured prober): tie on position and attitude — looped predictor does not improve metric groundability.

See [note.md](note.md) for the leak discovery, SkyJEPA comparison, and open questions.
