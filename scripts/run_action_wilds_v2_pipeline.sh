#!/usr/bin/env bash
# Full Wilds action v2 pipeline: fine-tune → protocol-B eval → residual.
# Intended to run under caffeinate/nohup (see launch via Terminal.app).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-aerojepa}"
PY="${PY:-$ROOT/.venv/bin/python}"
LOG="$ROOT/logs/pipeline_action_wilds_v2.log"
mkdir -p "$ROOT/logs" "$ROOT/results" "$ROOT/checkpoints"

exec > >(tee -a "$LOG") 2>&1
echo "=== PIPELINE START $(date -Iseconds) ==="

echo "=== 1/3 train action_conditioned_wilds_v2 ==="
"$PY" scripts/train.py --config configs/aerojepa_finetune_action_v2.yaml --device mps

echo "=== 2/3 evaluate_real ==="
"$PY" scripts/evaluate_real.py \
  --checkpoint checkpoints/action_conditioned_wilds_v2/latest.pt \
  --data-dir data/flights_128 --max-batches 8 \
  --out results/action_conditioned_wilds_v2_real_eval.json

echo "=== 3/3 residual ==="
"$PY" scripts/train_action_residual.py \
  --checkpoint checkpoints/action_conditioned_wilds_v2/latest.pt \
  --epochs 20 --num-train 384 --num-val 64 --batch-size 8 \
  --wind-fraction 0.4 --kick-fraction 0.2 --turn-fraction 0.2 \
  --wind-mps 2.0 --wind-mps-max 4.0 \
  --output-dir checkpoints/action_residual_wilds_v2 \
  --device mps

echo "=== PIPELINE_DONE $(date -Iseconds) ==="
