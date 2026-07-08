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

## 6. Interactive demo

```bash
python app.py --checkpoint checkpoints/world_model/latest.pt   # http://127.0.0.1:7860
```

## Determinism notes

- Seeds are set for Python, NumPy, and Torch (`utils/seed.py`); synthetic clips
  are deterministic functions of their per-sample seed.
- Exact floating-point results vary across hardware and backends (MPS vs CUDA vs
  CPU); trends and relative comparisons are stable, absolute values may drift
  slightly.
