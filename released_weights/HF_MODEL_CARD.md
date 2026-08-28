---
license: mit
library_name: pytorch
tags:
  - jepa
  - world-model
  - quadrotor
  - pytorch
---

# AeroJEPA — video world models for quadrotor egocentric dynamics

**Video-JEPA** on short egocentric drone clips (~3–5M params). Closest paper is
[SkyJEPA](https://arxiv.org/abs/2606.23444) (Rao et al., 2026): SkyJEPA is a
*state*-history JEPA with outdoor MPPI; this project is *video*-JEPA on RGB clips,
not a SkyJEPA clone. We do not claim UAV autonomy or onboard flight control.

## Weight files

| File | Registry key | Notes |
| --- | --- | --- |
| `world_model.pt` | `world_model` | Synthetic future-frame world model; Gradio default |
| `real_finetune_fast.pt` | `real_finetune_fast` | Unconditioned Wilds fine-tune; representation / gap tables |

Release checkpoints are `{config, model}` only (CPU tensors). Full training
checkpoints with optimizer state are not published.

**Not released:** action-conditioned Wilds / residuals (`action_conditioned_wilds`,
`action_residual_*`, `*_v2`). Counterfactual true/zero/shuffle tests fail on the
AC checkpoint, so it stays out of this host.

## Honest limits

- Short-horizon metric probing: **0.1 s**, \(\Delta t = 0.025\,\mathrm{s}\).
- Closed-loop L-turn stress: scale **×1.25 → 0%** success (10 seeds).
- Not a flight controller. Research demo only.

## Load

```python
import torch
from aerojepa.eval import load_pretrained

model, cfg = load_pretrained("world_model", torch.device("cpu"))
```

Or download into the GitHub layout:

```bash
./scripts/download_weights.sh world_model
```

**Code:** [github.com/JMangold0352/aerojepa](https://github.com/JMangold0352/aerojepa) · **License:** MIT
