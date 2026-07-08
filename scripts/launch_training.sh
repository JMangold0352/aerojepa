#!/usr/bin/env bash
# Launch a long training run from macOS Terminal.app (survives Cursor/IDE restarts).
#
# Uses caffeinate to prevent sleep and nohup so the shell can close safely.
# Logs are timestamped under logs/; a PID file tracks the active job.
#
# Usage:
#   ./scripts/launch_training.sh configs/aerojepa_finetune.yaml
#   ./scripts/launch_training.sh configs/aerojepa_finetune.yaml --device mps
#   ./scripts/launch_training.sh configs/aerojepa_finetune.yaml \
#       --resume checkpoints/real_finetune/latest.pt
#
# Monitor:
#   tail -f logs/train_<config>_<timestamp>.log
#   cat logs/train_<config>.pid
#
# Stop:
#   kill "$(cat logs/train_<config>.pid)"
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [-- extra train.py args]" >&2
  echo "Example: $0 configs/aerojepa_finetune.yaml --resume checkpoints/real_finetune/latest.pt" >&2
  exit 1
fi

CONFIG="$1"
shift

if [[ ! -f "$CONFIG" ]]; then
  echo "Config not found: $CONFIG" >&2
  exit 1
fi

PY="${PY:-$ROOT/.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
mkdir -p "$LOG_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-aerojepa}"

CFG_NAME="$(basename "$CONFIG" .yaml)"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="$LOG_DIR/train_${CFG_NAME}_${STAMP}.log"
PIDFILE="$LOG_DIR/train_${CFG_NAME}.pid"
LATEST_LOG_LINK="$LOG_DIR/train_${CFG_NAME}_latest.log"

# If caller did not pass --device, inject the default.
EXTRA_ARGS=("$@")
has_device=false
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--device" ]]; then
    has_device=true
    break
  fi
done

CMD=("$PY" scripts/train.py --config "$CONFIG")
if [[ "$has_device" == false ]]; then
  CMD+=(--device "$DEVICE")
fi
CMD+=("${EXTRA_ARGS[@]}")

{
  echo "=== AeroJEPA training launch $(date -Iseconds) ==="
  echo "config:  $CONFIG"
  echo "command: ${CMD[*]}"
  echo "log:     $LOG"
} | tee "$LOG"

caffeinate -dims nohup "${CMD[@]}" >>"$LOG" 2>&1 &
echo $! >"$PIDFILE"
ln -sf "$(basename "$LOG")" "$LATEST_LOG_LINK"

echo "Started PID $(cat "$PIDFILE")"
echo "Tail log: tail -f $LOG"
echo "         (or tail -f $LATEST_LOG_LINK)"
