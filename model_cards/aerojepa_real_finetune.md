# Model card: AeroJEPA real fine-tune (Wilds)

## Overview

Unconditioned world-model fine-tune on **The Wilds Drones** Parrot clips. Best
**representation** checkpoint for sim-to-real tables - not the closed-loop default
(that needs action conditioning; see
[aerojepa_action_wilds](aerojepa_action_wilds.md)).

- **Init:** synthetic / earlier real fine-tune chain ending at
  `checkpoints/real_finetune/latest.pt`
- **Config:** [`configs/aerojepa_finetune_fast.yaml`](../configs/aerojepa_finetune_fast.yaml)
- **Checkpoint:** `checkpoints/real_finetune_fast/latest.pt`

## Data & resolution

- Source folder: `data/flights_128/` (15 clips, square **128×128** on disk)
- Model input: **`img_size=64`** (inherited). Frames are resized at load time.
- See [`docs/EVAL_PROTOCOL.md`](../docs/EVAL_PROTOCOL.md).

## Training

| Knob | Value |
| --- | --- |
| Epochs | 10 |
| LR | 5e-5 |
| Window mode | sliding, stride 8 |
| Objective | future (world model), **no** action conditioning |

## Performance (protocol B)

```bash
python scripts/evaluate_real.py \
  --checkpoint checkpoints/real_finetune_fast/latest.pt \
  --data-dir data/flights_128 --max-batches 8 \
  --out results/real_finetune_fast_eval.json
```

| Metric | Value |
| --- | ---: |
| Real latent cosine | **0.974** |
| Synthetic latent cosine | **0.994** |
| Gap (synth − real) | **+0.019** |
| Real rollout @ h=4 | **0.961** |

Source: [`results/real_finetune_fast_eval.json`](../results/real_finetune_fast_eval.json).

## Intended use

- Headline sim-to-real / transfer numbers
- Open-loop planner demos that do not need 6-DoF conditioning
- Init for Tello fine-tune once personal footage exists

## Out of scope

- Closed-loop gradient planning with residuals (use action Wilds stack)
- Claiming “trained at 128×128” - storage resolution ≠ model input
