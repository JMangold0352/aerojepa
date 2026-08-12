#!/usr/bin/env bash
# After action_conditioned_wilds_v2 finishes training: protocol-B eval + residual.
# Waits on logs/train_action_wilds_v2.pid (or a PID passed as $1).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${PYTHONPATH:-}:$ROOT/src"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-aerojepa}"
PY="${PY:-$ROOT/.venv/bin/python}"
LOG="$ROOT/logs/train_action_wilds_v2_followup.log"
PIDFILE="$ROOT/logs/train_action_wilds_v2.pid"
WAIT_PID="${1:-}"
if [[ -z "$WAIT_PID" && -f "$PIDFILE" ]]; then
  WAIT_PID="$(cat "$PIDFILE")"
fi

{
  echo "=== followup start $(date -Iseconds) ==="
  if [[ -n "$WAIT_PID" ]]; then
    echo "waiting for train pid $WAIT_PID"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
    echo "train pid $WAIT_PID exited"
  fi
  test -f checkpoints/action_conditioned_wilds_v2/latest.pt

  echo "=== evaluate_real ==="
  "$PY" scripts/evaluate_real.py \
    --checkpoint checkpoints/action_conditioned_wilds_v2/latest.pt \
    --data-dir data/flights_128 --max-batches 8 \
    --out results/action_conditioned_wilds_v2_real_eval.json

  echo "=== residual ==="
  "$PY" scripts/train_action_residual.py \
    --checkpoint checkpoints/action_conditioned_wilds_v2/latest.pt \
    --epochs 20 --num-train 384 --num-val 64 --batch-size 8 \
    --wind-fraction 0.4 --kick-fraction 0.2 --turn-fraction 0.2 \
    --wind-mps 2.0 --wind-mps-max 4.0 \
    --output-dir checkpoints/action_residual_wilds_v2 \
    --device mps

  echo "=== FOLLOWUP_DONE $(date -Iseconds) ==="
} >>"$LOG" 2>&1
