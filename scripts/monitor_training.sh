#!/usr/bin/env bash
# After INITIAL_WAIT (default 10h), check training every INTERVAL (default 3.5h).
# Emits AGENT_LOOP_TICK_TRAIN for Cursor wake notifications.
set -euo pipefail
cd "$(dirname "$0")/.."
INITIAL_WAIT="${INITIAL_WAIT:-36000}"   # 10 hours
INTERVAL="${INTERVAL:-12600}"         # 3.5 hours
LOG="logs/train_all_day.log"
PIDFILE="logs/train_all_day.pid"
STATUS="logs/monitor_status.txt"

check_status() {
  local ts msg
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ -f "$LOG" ]] && grep -q "TRAIN_ALL_DAY_DONE" "$LOG" 2>/dev/null; then
    msg="DONE - all configs finished"
  elif [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    msg="RUNNING - pid $(cat "$PIDFILE")"
  elif pgrep -f "scripts/train.py" >/dev/null 2>&1; then
    msg="RUNNING - train.py active"
  else
    msg="STOPPED or not started"
  fi
  local last
  last="$(tail -3 "$LOG" 2>/dev/null | tr '\n' ' ' || echo 'no log yet')"
  echo "$ts | $msg | tail: $last" | tee -a "$STATUS"
  echo "$msg"
}

# First check immediately (training just started)
state="$(check_status)"
echo "AGENT_LOOP_TICK_TRAIN {\"prompt\":\"Check AeroJEPA training in ~/Projects/aerojepa. Read logs/train_all_day.log and logs/monitor_status.txt. State: $state. If TRAIN_ALL_DAY_DONE, run evaluate on all checkpoints and summarize. If stopped unexpectedly, report last error. If still running, brief progress only.\"}"

sleep "$INITIAL_WAIT"

while true; do
  state="$(check_status)"
  echo "AGENT_LOOP_TICK_TRAIN {\"prompt\":\"Check AeroJEPA training in ~/Projects/aerojepa. Read logs/train_all_day.log and logs/monitor_status.txt. State: $state. If TRAIN_ALL_DAY_DONE, run evaluate on all checkpoints and summarize. If stopped unexpectedly, report last error. If still running, brief progress only.\"}"
  if [[ "$state" == DONE* ]]; then
  echo "[$(date)] training finished; monitor exiting" >> "$STATUS"
    exit 0
  fi
  sleep "$INTERVAL"
done
