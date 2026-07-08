#!/usr/bin/env bash
# Train the main AeroJEPA recipes back-to-back. Logs to logs/train_all_day.log
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-aerojepa}"
PY="${PY:-.venv/bin/python}"
DEVICE="${DEVICE:-auto}"
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_all_day.log"
PIDFILE="$LOG_DIR/train_all_day.pid"
echo $$ > "$PIDFILE"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== AeroJEPA all-day training started (device=$DEVICE) ==="
log "PID $$"

CONFIGS=(
  configs/aerojepa_baseline.yaml
  configs/aerojepa_looped.yaml
  configs/aerojepa_world_model.yaml
  configs/aerojepa_action_conditioned.yaml
)

for cfg in "${CONFIGS[@]}"; do
  log "--- training: $cfg ---"
  # Append directly to the log (no tee pipe) so a broken pipe cannot kill training.
  "$PY" scripts/train.py --config "$cfg" --device "$DEVICE" >>"$LOG" 2>&1
  log "--- finished: $cfg ---"
done

log "=== All training complete ==="
rm -f "$PIDFILE"
echo "TRAIN_ALL_DAY_DONE $(date -Iseconds)" >> "$LOG"
