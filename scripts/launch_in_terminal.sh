#!/usr/bin/env bash
# Run from macOS Terminal.app (not Cursor) for overnight training that survives IDE restarts.
#   cd ~/Projects/aerojepa && ./scripts/launch_in_terminal.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-aerojepa}"
mkdir -p logs
caffeinate -dims nohup ./scripts/train_all_day.sh >> logs/train_nohup.out 2>&1 &
echo "Started PID $! — log: logs/train_all_day.log"
