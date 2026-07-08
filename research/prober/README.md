# AeroProber (LoopedJEPA-Prober)

A physics-inspired prober that converts latent rollouts from a frozen AeroJEPA
world model into physically meaningful metric states (position, velocity, Euler
attitude, angular velocity) through a differentiable kinematic integrator.

## Quick start

```bash
# From the repo root. PyFlyt must run OUTSIDE the Cursor sandbox.
# 1. Train the structured prober (regular predictor baseline):
python research/prober/scripts/train_prober.py \
    --config research/prober/configs/prober_synth.yaml

# 2. Run the headline ablation (5 seeds, naive/plain/structured):
python research/prober/scripts/run_ablations.py \
    --config research/prober/configs/prober_synth.yaml \
    --seeds 0 1 2 3 4 --num-train 256 --epochs 30

# 3. Evaluate on real Parrot Wilds footage:
python research/prober/scripts/eval_real.py \
    --prober research/prober/results/prober_regular/best.pt \
    --data-dir data/flights_with_state
```

## Layout

```
research/prober/
  src/aerojepa_research/prober/
    integrator.py     differentiable Euler-angle kinematic integrator
    prober.py         Prober MLP (~5k params) + PlainMLPHead ablation arm
    data_pyflyt.py    PyFlyt clip generator (frames + actions + metric state)
    rollout.py        frozen-model rollout extraction (regular + looped)
    loss.py           structured / plain / naive loss fns (wrapped-angle att)
    metrics.py        position/velocity/attitude RMSE + per-horizon curves
    wilds_state.py    extended Parrot Wilds converter (preserves abs state)
  scripts/
    train_prober.py   config-driven trainer (--arm structured|plain|naive)
    run_ablations.py  headline experiment (multi-seed, paired, figures)
    eval_real.py      real-data eval on Wilds (velocity/attitude/altitude RMSE)
  tests/              20 tests (unit + integration)
  configs/prober_synth.yaml
  results/            JSON metrics + figures (gitignored except JSON)
  note.md             technical note for PhD review
```

## Design decisions (v1)

- **Attitude:** Euler angles (yaw, pitch, roll), wrapped to (-180, 180].
- **Ground truth:** PyFlyt quadrotor simulator (real dynamics, full 6-DoF).
- **Frozen:** encoder + predictor + EMA teacher. Only the prober trains.
- **Phasing:** regular (single-pass) predictor first; looped arm added as a
  comparison (the project's headline research question).
- **Real data:** Parrot Wilds, quantitative for velocity/attitude/altitude only
  (no GPS -> position x/y is dead-reckoned, qualitative).

See [note.md](note.md) for the full comparison to SkyJEPA and open questions.
