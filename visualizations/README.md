# Visualizations

Every figure is generated from a trained checkpoint, so nothing here is
hand-drawn or cherry-picked. Regenerate the whole set with:

```bash
python scripts/visualize.py --checkpoint checkpoints/looped/latest.pt
# quick smoke render (fewer clips):
python scripts/visualize.py --checkpoint checkpoints/looped/latest.pt --fast
```

Output lands in `visualizations/figures/` (git-ignored; regenerate as needed).

### Per-checkpoint figure suite

```bash
python scripts/visualize.py --checkpoint checkpoints/looped/latest.pt
```

### Ablation comparison figures

```bash
python scripts/run_ablations.py --mode quick    # or --mode full
python visualizations/compare_ablations.py
# -> visualizations/figures/ablations/
```

| Figure | Reads as |
| --- | --- |
| `ablations/01_latent_cosine_bar.png` | Headline latent cosine across ablation variants. |
| `ablations/02_smooth_l1_bar.png` | Complementary smooth-L1 distance (lower is better). |
| `ablations/03_per_loop_cosine.png` | Refinement gain per loop - the curve that separates variants at 20 ep. |
| `ablations/04_rollout_comparison.png` | Rollout cosine vs horizon, all variants on one axis. |
| `ablations/05_rollout_panel.png` | Side-by-side rollout small multiples. |
| `ablations/06_rollout_comparison.gif` | Animated build-up of rollout curves. |

### Single-checkpoint suite

| Figure | Reads as |
| --- | --- |
| `00_clip.png` | A held-out synthetic drone clip -- what the model actually sees. |
| `01_rollout.png` | Prediction quality vs how far ahead we forecast. A gentle decline is a healthy world model. |
| `02_per_loop_cosine.png` | Prediction quality after each refinement loop. Rising = recurrence is doing real work. |
| `03_exit_distribution.png` | How the learned exit gate spends compute across depths. |
| `04_latent_trajectory.png` | The clip's motion traced through latent space (2D PCA). |
| `05_attention.png` | Predictor self-attention with frame boundaries -- does it reason across time? |

Figures use a shared 300-DPI style defined in
[`src/aerojepa/viz/style.py`](../src/aerojepa/viz/style.py).
