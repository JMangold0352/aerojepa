# Ablation summary

Source: `results/ablations/summary.json`
Mode: **quick** · Epochs: **20**

| Variant | Objective | Latent cosine | Smooth-L1 | Rollout @ last h |
| --- | --- | ---: | ---: | ---: |
| baseline | masked | 0.9933 | 0.0069 | 0.9900 |
| loops_2 | masked | 0.9927 | 0.0075 | 0.9893 |
| loops_3 | masked | 0.9931 | 0.0071 | 0.9898 |
| world_model | future | 0.9943 | 0.0059 | 0.9923 |

## Figures

- `01_latent_cosine_bar.png`
- `02_smooth_l1_bar.png`
- `03_per_loop_cosine.png`
- `04_rollout_comparison.png`
- `05_rollout_panel.png`
- `06_rollout_comparison.gif`
