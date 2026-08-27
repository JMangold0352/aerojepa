# Scripts

Every command-line entry point for AeroJEPA. All scripts add `src/` to the path
automatically, so they run from a plain checkout (no `pip install -e .` needed).

| Script | What it does |
| --- | --- |
| `verify_install.py` | Fast environment + code sanity check (render clip, one forward/backward pass). Start here. |
| `generate_synthetic.py` | Render preview contact sheets of the synthetic drone clips. |
| `train.py` | Train a model from a config: `--config configs/aerojepa_looped.yaml`. |
| `evaluate.py` | Full metrics report for a checkpoint (latent quality, rollout, loop analysis). |
| `compare_baseline.py` | Head-to-head latent-prediction comparison of two checkpoints. |
| `run_ablations.py` | Train + score variant suite (`--mode quick` 20ep / `--mode full` 100ep). |
| `visualize.py` | Regenerate the publication figure suite from a checkpoint. |
| `run_planner_demo.py` | Latent-space planner demo (GIF + trajectory plot). |
| `run_closed_loop_demo.py` | Closed-loop PyFlyt hover / waypoint / recover vs baselines. `--planner {shooting,gradient}`, `--residual-checkpoint`. |
| `train_action_residual.py` | Train a tiny learned residual on top of the frozen LatentPlanner action map. |
| `compare_action_residual.py` | Closed-loop before/after metrics: heuristic map vs residual. |
| `compare_planner_modes.py` | Closed-loop drift: random-shooting vs gradient multi-step planning (recover). |
| `compare_full_stack.py` | Multi-seed closed-loop: hover / residual / full stack. See `docs/EVAL_PROTOCOL.md`. |
| `run_stress_suite.py` | Wind-gust + aggressive L-turn stress tests; writes GIFs, metrics, breaking-points summary. |
| `stitch_closed_loop_demo.py` | Stitch closed-loop GIFs into `docs/gallery/closed_loop_demo_reel.gif`. |
| `capture_tello.py` | Record a DJI Tello clip + telemetry to `data/flights/` (record-only, never flies). `--preflight`, `--duration`, `--fps`. |
| `preprocess_real.py` | Standardize any footage into the training format; `--probe` inspects a folder (dataset doctor). |
| `convert_wilds.py` | Convert The Wilds Drones (Parrot JSON + MP4) into `data/flights/` with telemetry CSVs. |
| `tello_workflow.sh` | **Tello end-to-end:** preflight → tagged sessions → preprocess → fine-tune → transfer report. |
| `tello_compare_report.py` | Compare Wilds-only vs Tello-fine-tuned checkpoints on personal footage. |
| `run_transfer_curve.py` | Sim-to-real transfer curve vs real-data volume (1 / 5 / 15 clips). |
| `evaluate_real.py` | Compare synthetic-vs-real metrics for a checkpoint (sim-to-real gap). |
| `launch_training.sh` | **Long runs from Terminal.app** - caffeinate + nohup + timestamped logs. Supports `--resume`. |
| `launch_in_terminal.sh` | Train all four synthetic recipes back-to-back (legacy all-day script). |
| `gradio_demo.py` | Launch the interactive demo (same as `python app.py`). |

## Real-data workflow

```bash
# Personal Tello footage (after Wilds fine-tune exists):
./scripts/tello_workflow.sh all --duration 30

python scripts/capture_tello.py --preflight              # check link/battery (no flight)
python scripts/capture_tello.py --duration 30 --fps 15   # record while a pilot flies
python scripts/preprocess_real.py --probe --input-dir data/flights
python scripts/preprocess_real.py --input-dir data/raw --output-dir data/flights --square
./scripts/launch_training.sh configs/aerojepa_finetune.yaml
# Resume after interruption:
./scripts/launch_training.sh configs/aerojepa_finetune.yaml --resume checkpoints/real_finetune/latest.pt
python scripts/evaluate_real.py --checkpoint checkpoints/real_finetune/latest.pt
```

## Typical workflow

```bash
python scripts/verify_install.py
python scripts/train.py --config configs/aerojepa_baseline.yaml
python scripts/train.py --config configs/aerojepa_looped.yaml
python scripts/compare_baseline.py \
  --baseline-checkpoint checkpoints/baseline/latest.pt \
  --looped-checkpoint  checkpoints/looped/latest.pt
python scripts/evaluate.py --checkpoint checkpoints/looped/latest.pt
python scripts/visualize.py --checkpoint checkpoints/looped/latest.pt
```
