# Results

JSON metrics from evaluation runs. Regenerate with:

```bash
./scripts/evaluate_all.sh
```

## Synthetic benchmark (100 epochs, MPS)

| Model | Latent cosine | Rollout h=4 | Loops | File |
| --- | ---: | ---: | ---: | --- |
| baseline | 0.954 | 0.917 | — | [`baseline_eval.json`](baseline_eval.json) |
| looped | 0.961 | 0.925 | 1.50 | [`looped_eval.json`](looped_eval.json) |
| world_model | 0.981 | 0.973 | 1.75 | [`world_model_eval.json`](world_model_eval.json) |
| action_conditioned | 0.980 | 0.975 | 1.75 | [`action_conditioned_eval.json`](action_conditioned_eval.json) |

Head-to-head: [`comparison.json`](comparison.json) (baseline vs looped, +0.007 cosine).

## Ablations

Controlled variant suite on the same synthetic recipe.

| Command | Epochs | Purpose |
| --- | ---: | --- |
| `python scripts/run_ablations.py --mode quick` | 20 | Fast iteration signal |
| `python scripts/run_ablations.py --mode full` | 100 | Publication-quality table |
| `python scripts/run_ablations.py --eval-only` | — | Re-score existing `checkpoints/ablations/` |

Summary: [`ablations/summary.json`](ablations/summary.json) (includes rollout + per-loop metrics).
Per-variant: `ablations/{baseline,loops_2,loops_3,world_model}.json`.

Figures: [`../visualizations/figures/ablations/`](../visualizations/figures/ablations/) —
regenerate with `python visualizations/compare_ablations.py`.

## Transfer curve (sim-to-real vs data volume)

Fine-tune `world_model` on 1 / 5 / 15 Wilds clips (3 held out for eval); plot gap + rollout.

```bash
python scripts/run_transfer_curve.py --device mps
python visualizations/plot_transfer_curve.py   # re-render figure only
```

| Artifact | Contents |
| --- | --- |
| [`transfer_curve/summary.json`](transfer_curve/summary.json) | All points + metadata |
| [`transfer_curve/manifest.json`](transfer_curve/manifest.json) | Train/eval split |
| [`transfer_curve/transfer_curve.png`](transfer_curve/transfer_curve.png) | Publication figure |

## Figures (main models)

- [`../visualizations/figures/looped/`](../visualizations/figures/looped/)
- [`../visualizations/figures/world_model/`](../visualizations/figures/world_model/)
