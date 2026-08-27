# Closed-loop stress suite - breaking points

**Checkpoint (baseline map):** `checkpoints/action_conditioned/latest.pt`  
**Artifacts:** `visualizations/closed_loop/stress/`

## Tasks

| Task | Stressor | Success criterion |
| --- | --- | --- |
| `wind_gust` | Constant lateral PyFlyt wind (default **2 m/s**) after settle | Survive; max XY drift ≤ 1.5 m |
| `aggressive_turn` | L-course with a sharp 90° corner | Clear both legs without crash/dome exit |

## Baseline breaks (heuristic map, shooting planner)

### Wind (2 m/s, 200 steps)

| Policy | max XY | Failure |
| --- | ---: | --- |
| planner | 1.54 m | `excessive_drift` |
| hover | 0.79 m | `ok` |
| random | 6.23 m | `crash` |

Planner survived but drifted more than doing nothing - over-correction under sustained wind.

### Aggressive turn (hard 0.8 m legs, early runs)

| Policy | Legs | Failure |
| --- | --- | --- |
| planner | 1/2 | `out_of_bounds` / crash at corner |
| seek | 1/2 | `crash` |
| hover | 0/2 | `missed_turn` |

Everyone who attempted the turn failed at the corner.

## After residual + gradient + latent refine

Wind-augmented / multi-stress residual + gradient planner. Numbers from
`visualizations/closed_loop/full_stack_compare*.json`. Preferred stack:
`action_conditioned_wilds` + `action_residual_wilds` (see
[`docs/EVAL_PROTOCOL.md`](../../../docs/EVAL_PROTOCOL.md)).

| Task | Baseline ok | Full stack ok | Notes |
| --- | ---: | ---: | --- |
| wind_gust (2 m/s) | 0% | 100% | Full stack beats hover max XY |
| aggressive_turn (hard 0.8 m) | 0% | 100% | Soft course also improved |
| recover | - | 100% | Adaptive brake after kick (shared) |
| hover | 100% | 100% | Tighter station-keeping with residual |

Recover survival is mostly the brake; the planner’s edge is recovery speed.

## Reproduce

```bash
# Baseline stress:
python scripts/run_stress_suite.py \
  --checkpoint checkpoints/action_conditioned/latest.pt --wind-mps 2.0

# Full stack:
python scripts/run_stress_suite.py \
  --residual-checkpoint checkpoints/action_residual_wind/best.pt \
  --planner gradient --latent-smooth 0.05 \
  --out-dir visualizations/closed_loop/stress_full_stack

# Multi-seed table:
python scripts/compare_full_stack.py --seeds 0 1 2 3 4 --include-hard-turn
```
