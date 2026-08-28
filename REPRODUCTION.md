# Reproduction guide

Everything below runs from a clean checkout with **no data downloads** -- the
synthetic drone generator supplies the clips. No GPU is required (Apple Silicon
MPS, CUDA, or CPU are all auto-detected).

## 0. Environment

```bash
git clone <your-fork-url> aerojepa && cd aerojepa
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # core
pip install -r requirements-dev.txt      # + tests, demo, viz, video
```

Sanity check:

```bash
python scripts/verify_install.py
pytest -q
```

## 1. Inspect the data

```bash
python scripts/generate_synthetic.py --num-clips 3
# -> results/synthetic_preview/clip_00.png ...
```

## 2. Fast end-to-end smoke run (~1 minute, CPU)

```bash
python scripts/train.py --config configs/smoke_test.yaml
python scripts/evaluate.py --checkpoint checkpoints/smoke/latest.pt
```

This exercises the entire pipeline: data -> model -> loss -> checkpoint -> eval.

## 3. Full training recipes

```bash
python scripts/train.py --config configs/aerojepa_baseline.yaml
python scripts/train.py --config configs/aerojepa_looped.yaml
python scripts/train.py --config configs/aerojepa_world_model.yaml
python scripts/train.py --config configs/aerojepa_action_conditioned.yaml
```

Checkpoints are written to `checkpoints/<name>/latest.pt`; metrics stream to
`runs/<name>/metrics.jsonl`.

Tune cost vs quality in the config: `data.num_train`, `train.epochs`,
`data.img_size`, `encoder.depth`, and `predictor.max_loops` are the main knobs.

## 4. Compare, evaluate, and visualize

```bash
python scripts/compare_baseline.py \
  --baseline-checkpoint checkpoints/baseline/latest.pt \
  --looped-checkpoint  checkpoints/looped/latest.pt

python scripts/evaluate.py --checkpoint checkpoints/looped/latest.pt
# -> results/looped_eval.json

python scripts/visualize.py --checkpoint checkpoints/looped/latest.pt
# -> visualizations/figures/*.png
```

## 5. Ablation suite

```bash
python scripts/run_ablations.py --mode quick    # 20 epochs
# or: python scripts/run_ablations.py --mode full   # 100 epochs
python visualizations/compare_ablations.py
# -> results/ablations/summary.json
# -> visualizations/figures/ablations/
```

## 6. Download released weights

Hosted files: [`released_weights/`](released_weights/). Until
`released_weights/urls.yaml` has real URLs (not `PLACEHOLDER_URL`), download
fails — train locally instead.

```bash
pip install -e ".[hf]"                 # optional
./scripts/download_weights.sh --list
./scripts/download_weights.sh world_model
# -> checkpoints/world_model/latest.pt
```

```python
import torch
from aerojepa.eval import load_pretrained
model, cfg = load_pretrained("world_model", torch.device("cpu"))
```

Action-conditioned Wilds is not released (counterfactuals fail).

## 7. Interactive demo

```bash
./scripts/download_weights.sh world_model
python app.py --checkpoint checkpoints/world_model/latest.pt   # http://127.0.0.1:7860
```

## 8. Closed-loop (PyFlyt)

Needs `pip install -e ".[sim]"`. Run outside sandboxed environments.

```bash
python scripts/run_closed_loop_demo.py \
  --checkpoint checkpoints/action_conditioned_wilds/latest.pt \
  --residual-checkpoint checkpoints/action_residual_wilds/best.pt \
  --planner gradient --latent-smooth 0.05 --task hover
```

Full metric recipes (including 64 vs 128, gap eval, multi-seed tables):
[`docs/EVAL_PROTOCOL.md`](docs/EVAL_PROTOCOL.md).

## Determinism notes

- Seeds are set for Python, NumPy, and Torch (`utils/seed.py`); synthetic clips
  are deterministic functions of their per-sample seed.
- Exact floating-point results vary across hardware and backends (MPS vs CUDA vs
  CPU); trends and relative comparisons are stable, absolute values may drift
  slightly.

## Continuous integration

The repo includes [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (lint /
tests). If `git push` rejects the workflow file with a `workflow` scope error,
refresh credentials then push again:

```bash
gh auth refresh -s workflow
git push
```

Until that succeeds, CI may exist only in the local tree and not yet on GitHub.
