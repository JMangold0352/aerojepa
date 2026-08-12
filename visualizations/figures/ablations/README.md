# Ablation summary

Source: `results/ablations/summary.json`
Mode: **full** · Epochs: **100**

| Variant | Objective | Latent cosine | Smooth-L1 | Rollout @ last h |
| --- | --- | ---: | ---: | ---: |
| baseline | masked | 0.9533 | 0.0417 | 0.9171 |
| loops_2 | masked | 0.9600 | 0.0359 | 0.9249 |
| loops_3 | masked | 0.9628 | 0.0334 | 0.9279 |
| world_model | future | 0.9813 | 0.0174 | 0.9730 |

## Figures

- `01_latent_cosine_bar.png`
- `02_smooth_l1_bar.png`
- `03_per_loop_cosine.png`
- `04_rollout_comparison.png`
- `05_rollout_panel.png`
- `06_rollout_comparison.gif`
