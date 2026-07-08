# README gallery assets

Tracked copies of key figures embedded in the root [README.md](../../README.md).
Regenerate from trained checkpoints, then refresh this folder:

```bash
# Real-data best model
python scripts/visualize.py --checkpoint checkpoints/real_finetune_fast/latest.pt \
  --out-dir visualizations/figures/real_finetune_fast
python scripts/run_planner_demo.py --checkpoint checkpoints/real_finetune_fast/latest.pt --task hover
python scripts/run_planner_demo.py --checkpoint checkpoints/real_finetune_fast/latest.pt --task waypoint

cp visualizations/figures/real_finetune_fast/01_rollout.png docs/gallery/real_rollout.png
cp visualizations/figures/real_finetune_fast/02_per_loop_cosine.png docs/gallery/real_per_loop_cosine.png
cp visualizations/planner/real_finetune_fast/plan_hover.gif docs/gallery/planner_hover_real.gif
cp visualizations/planner/real_finetune_fast/plan_waypoint.gif docs/gallery/planner_waypoint_real.gif

# Synthetic pretrain
python scripts/visualize.py --checkpoint checkpoints/world_model/latest.pt \
  --out-dir visualizations/figures/world_model
python scripts/visualize.py --checkpoint checkpoints/looped/latest.pt \
  --out-dir visualizations/figures/looped
python scripts/run_planner_demo.py --checkpoint checkpoints/world_model/latest.pt --task waypoint
python visualizations/compare_ablations.py

cp visualizations/figures/world_model/01_rollout.png docs/gallery/world_model_rollout.png
cp visualizations/figures/looped/02_per_loop_cosine.png docs/gallery/looped_per_loop_cosine.png
cp visualizations/figures/looped/05_attention.png docs/gallery/looped_attention.png
cp visualizations/planner/plan_waypoint.gif docs/gallery/planner_waypoint.gif
cp visualizations/figures/ablations/03_per_loop_cosine.png docs/gallery/ablation_per_loop.png
cp visualizations/figures/ablations/04_rollout_comparison.png docs/gallery/ablation_rollout.png
```

| File | Source |
| --- | --- |
| `real_rollout.png` | Real-data rollout cosine vs horizon |
| `real_per_loop_cosine.png` | Real-data per-loop refinement |
| `planner_hover_real.gif` | Latent planner on real-data checkpoint (hover) |
| `planner_waypoint_real.gif` | Latent planner on real-data checkpoint (waypoint) |
| `world_model_rollout.png` | Synthetic rollout cosine vs horizon |
| `looped_per_loop_cosine.png` | Synthetic refinement gain per loop |
| `looped_attention.png` | Predictor attention across time |
| `planner_waypoint.gif` | Latent planner demo (synthetic checkpoint) |
| `ablation_per_loop.png` | Ablation per-loop comparison |
| `ablation_rollout.png` | Ablation rollout comparison |
| `transfer_curve.png` | Sim-to-real gap vs fine-tune clip count |
