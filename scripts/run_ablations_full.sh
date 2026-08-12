#!/usr/bin/env bash
# Wait for the Wilds v2 pipeline (if any), then run publication ablations.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-aerojepa}"
PY="${PY:-$ROOT/.venv/bin/python}"
LOG="$ROOT/logs/ablations_full.log"
mkdir -p "$ROOT/logs" "$ROOT/results/ablations"

exec > >(tee -a "$LOG") 2>&1
echo "=== ABLATIONS WAIT $(date -Iseconds) ==="

# Prefer waiting on the pipeline script; also clear residual/train if orphaned.
while pgrep -f "run_action_wilds_v2_pipeline|aerojepa_finetune_action_v2|train_action_residual.py.*wilds_v2" >/dev/null 2>&1; do
  echo "[ablations] MPS busy (Wilds v2 pipeline); sleep 60s — $(date -Iseconds)"
  sleep 60
done

echo "=== ABLATIONS START --mode full $(date -Iseconds) ==="
"$PY" scripts/run_ablations.py --mode full --device mps

echo "=== compare_ablations ==="
"$PY" visualizations/compare_ablations.py

echo "=== ABLATIONS_DONE $(date -Iseconds) ==="
