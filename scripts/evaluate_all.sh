#!/usr/bin/env bash
# Evaluate all shipped checkpoints and write results/*_eval.json.
# If real footage is present in data/flights/, also report the synthetic-vs-real
# gap for the world-model checkpoints via scripts/evaluate_real.py.
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-aerojepa}"
PY="${PY:-.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
REAL_DIR="${REAL_DIR:-data/flights}"

for ckpt in baseline looped world_model action_conditioned; do
  echo "=== $ckpt ==="
  "$PY" scripts/evaluate.py \
    --checkpoint "checkpoints/$ckpt/latest.pt" \
    --device "$DEVICE" \
    --max-batches 8 \
    --out "results/${ckpt}_eval.json"
done

# Real-data gap: only runs when clips exist and a fine-tuned/base checkpoint is present.
if compgen -G "$REAL_DIR/*.mp4" > /dev/null || compgen -G "$REAL_DIR/*.mov" > /dev/null; then
  for ckpt in world_model action_conditioned real_finetune real_world_model; do
    if [ -f "checkpoints/$ckpt/latest.pt" ]; then
      echo "=== $ckpt (synthetic vs real) ==="
      "$PY" scripts/evaluate_real.py \
        --checkpoint "checkpoints/$ckpt/latest.pt" \
        --data-dir "$REAL_DIR" \
        --device "$DEVICE" \
        --max-batches 8 \
        --out "results/${ckpt}_real_eval.json"
    fi
  done
else
  echo "No real clips in $REAL_DIR; skipping real-data gap (see data/README.md)."
fi

echo "Done. See results/README.md"
