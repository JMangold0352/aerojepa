# Model card: AeroJEPA-Base

## Overview

AeroJEPA-Base is a self-supervised **video representation learner**. It encodes
short drone clips into space-time latents by learning to predict the latent
content of hidden (masked) space-time regions from visible context, supervised by
an EMA teacher. It is the general-purpose backbone; the
[world-model card](aerojepa_world_model.md) covers the forward-prediction variant.

- **Objective:** masked latent prediction (`objective: masked`).
- **Inputs:** clips `(T, C, H, W)`; default `T=8`, `64x64`, RGB.
- **Outputs:** per-token latents; per-frame or pooled features for downstream use.
- **Config:** [`configs/aerojepa_baseline.yaml`](../configs/aerojepa_baseline.yaml)
  (feed-forward) and [`configs/aerojepa_looped.yaml`](../configs/aerojepa_looped.yaml)
  (recurrent predictor + exit gate + sandwich RMSNorm).

## Architecture

| Component | Setting (default) |
| --- | --- |
| Tokenizer | per-frame 2D patches (`patch=8` -> 64 spatial tokens/frame); tubelet optional |
| Position encoding | factorized spatial + temporal tables |
| Encoder | ViT, `embed_dim=192`, `depth=6`, `heads=3` |
| Target encoder | EMA copy, stop-gradient, frozen |
| Predictor | narrow ViT, `embed_dim=96`, `depth=4`; looped variant re-applies it up to 2x with a learned exit gate |
| Loss | smooth-L1 in latent space (+ exit-entropy for the looped variant) |

Parameter budget: single-digit-to-low-tens-of-millions of trainable parameters
(the EMA copy is not counted). Print the exact count with
`model.num_trainable_params()`.

## Intended use

- A frozen backbone for downstream drone-perception tasks with a small trained
  head (label-efficient transfer).
- A research substrate for studying recurrent predictors on video.

## Out-of-scope / cautions

- Not a detector or classifier on its own; it produces representations.
- Trained (in this card's numbers) on **synthetic** clips; check `REPORT.md` /
  `results/` for real-data evals.

## Performance

Regenerate on your hardware:

```bash
python scripts/train.py --config configs/aerojepa_looped.yaml
python scripts/evaluate.py --checkpoint checkpoints/looped/latest.pt
```

| Metric | baseline | looped | How |
| --- | ---: | ---: | --- |
| Latent cosine (synthetic val) | 0.954 | **0.961** | `results/*_eval.json` |
| vs baseline delta | — | **+0.007** | `results/comparison.json` |
| Per-loop cosine | — | 0.940 → 0.959 | `results/looped_eval.json` |
| Expected loops | — | 1.50 | exit gate |

## Load and run

```python
import torch
from aerojepa.eval import load_model

model, cfg = load_model("checkpoints/looped/latest.pt", torch.device("cpu"))
clip = torch.rand(1, cfg["data"]["num_frames"], 3, cfg["data"]["img_size"], cfg["data"]["img_size"])
tokens = model.encoder.forward_all_patches(clip)   # (1, T*S, D)
```

## Lineage

Extends [looped-jepa](https://github.com/JMangold0352/looped-jepa) (the recurrent
predictor, exit gate, and sandwich RMSNorm) into the temporal domain.
