# Model card: AeroJEPA action-conditioned Wilds + residual

## Overview

Action-conditioned world model fine-tuned on Wilds Parrot clips, plus a small
**ActionResidualHead** trained on PyFlyt (wind / kick / turn mix). This is the
**default closed-loop** stack.

| Piece | Path |
| --- | --- |
| World model | `checkpoints/action_conditioned_wilds/latest.pt` |
| Residual | `checkpoints/action_residual_wilds/best.pt` |
| WM config | [`configs/aerojepa_finetune_action.yaml`](../configs/aerojepa_finetune_action.yaml) |
| Continuation | [`configs/aerojepa_finetune_action_v2.yaml`](../configs/aerojepa_finetune_action_v2.yaml) |

## World model

- Init: `checkpoints/action_conditioned/latest.pt` (synthetic action-conditioned)
- Data: `data/flights_128/` → resized to **64×64**
- Keeps `action_conditioning: true` (6-DoF)

### Protocol B (representation)

```bash
python scripts/evaluate_real.py \
  --checkpoint checkpoints/action_conditioned_wilds/latest.pt \
  --data-dir data/flights_128 --max-batches 8 \
  --out results/action_conditioned_wilds_real_eval.json
```

| Metric | Value |
| --- | ---: |
| Real latent cosine | **0.957** |
| Synthetic latent cosine | **0.994** |
| Gap | **+0.037** |

Weaker than [`real_finetune_fast`](aerojepa_real_finetune.md) on protocol-B cosine.
**Action counterfactuals fail** (true≈zero≈shuffle cosine ≈0.994). Kept for the
closed-loop action interface.
`*_wilds_v2` is worse and not the default.

Model input **64×64** (Wilds clips may be 128 on disk). Cosine = predictor↔EMA teacher
alignment on held-out clips (training diagnostic).

## Residual

Frozen WM; train ~1-3k-param MLP to correct heuristic AeroJEPA→PyFlyt map.

```bash
python scripts/train_action_residual.py \
  --checkpoint checkpoints/action_conditioned_wilds/latest.pt \
  --epochs 10 --num-train 256 \
  --wind-fraction 0.4 --kick-fraction 0.2 --turn-fraction 0.2 \
  --output-dir checkpoints/action_residual_wilds
```

Last published val MSE ≈ **0.329** (vs heuristic ≈ 0.366) in
`checkpoints/action_residual_wilds/history.json`.

## Closed-loop

```bash
python scripts/run_closed_loop_demo.py \
  --checkpoint checkpoints/action_conditioned_wilds/latest.pt \
  --residual-checkpoint checkpoints/action_residual_wilds/best.pt \
  --planner gradient --latent-smooth 0.05 --task hover
```

Multi-seed (seeds 0-2): wind / soft turn / hard turn / recover / hover all
**100%** success -
[`visualizations/closed_loop/full_stack_compare_wilds.json`](../visualizations/closed_loop/full_stack_compare_wilds.json).
Protocol details: [`docs/EVAL_PROTOCOL.md`](../docs/EVAL_PROTOCOL.md).

## Limitations

- Research demo in PyFlyt, not a flight controller
- Recover survival shares adaptive braking across policies
- Soft L-turn on older synthetic stacks is less reliable than this Wilds stack
