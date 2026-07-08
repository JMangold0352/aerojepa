#!/usr/bin/env bash
# AeroJEPA — personal Tello footage pipeline (one-command friendly).
#
# Record-only capture (never flies the drone), preprocess, fine-tune from the
# Wilds-adapted checkpoint, and compare vs Wilds-only.
#
# Prerequisites:
#   pip install djitellopy opencv-python
#   DJI Tello powered on; Mac joined to Tello Wi-Fi (192.168.10.1)
#
# Quick start:
#   ./scripts/tello_workflow.sh all --duration 30
#
# Steps individually:
#   ./scripts/tello_workflow.sh preflight
#   ./scripts/tello_workflow.sh sessions --duration 30
#   ./scripts/tello_workflow.sh preprocess
#   ./scripts/tello_workflow.sh train
#   ./scripts/tello_workflow.sh report
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-aerojepa}"
PY="${PY:-.venv/bin/python}"
DEVICE="${DEVICE:-mps}"
DURATION="${DURATION:-30}"
MIN_BATTERY="${MIN_BATTERY:-20}"
YES="${YES:-0}"

RAW="data/raw/tello"
FLIGHTS="data/flights_tello"
PROCESSED="data/flights_tello_128"
WILDS="data/flights_128"
BASELINE_CKPT="checkpoints/real_finetune_fast/latest.pt"
TELLO_CKPT="checkpoints/real_finetune_tello/latest.pt"
TELLO_CONFIG="configs/aerojepa_finetune_tello.yaml"

# Maneuver bundle: prefix | pilot instruction
MANEUVERS=(
  "hover|Hover in place — minimal drift, steady altitude"
  "forward|Fly forward ~2 m, hold, then return toward start"
  "turn|Yaw left/right in place — smooth 90° turns"
  "altitude|Climb ~0.5 m, hold, descend slowly"
)

SAFETY_BANNER='
================================================================================
 TELLO WORKFLOW — SAFETY
 * This pipeline RECORDS ONLY. You fly manually (Tello app or controller).
 * Fly in a large open area; keep props clear of people, pets, and hands.
 * Join Tello Wi-Fi before preflight. Charge battery above the minimum.
 * Indoors: good lighting, minimal wind; watch downwash near walls/ceilings.
 * Ctrl-C stops recording cleanly; you remain responsible for airspace rules.
 * Telemetry: each clip gets .raw.csv (full log) + .csv (training deltas).
================================================================================
'

die() { echo "ERROR: $*" >&2; exit 1; }

need_py() {
  [[ -x "$PY" ]] || die "Python not found at $PY — create .venv or set PY=..."
}

need_baseline() {
  [[ -f "$BASELINE_CKPT" ]] || die "Missing $BASELINE_CKPT — train Wilds fine-tune first."
}

parse_yes_flag() {
  for arg in "$@"; do
    [[ "$arg" == "--yes" || "$arg" == "-y" ]] && YES=1
  done
}

parse_duration() {
  local args=("$@")
  local i=0
  while (( i < ${#args[@]} )); do
    case "${args[$i]}" in
      --duration)
        DURATION="${args[$((i + 1))]:-30}"
        i=$((i + 2))
        ;;
      --min-battery)
        MIN_BATTERY="${args[$((i + 1))]:-20}"
        i=$((i + 2))
        ;;
      *)
        i=$((i + 1))
        ;;
    esac
  done
}

confirm_safety() {
  if [[ "$YES" == "1" ]]; then
    return 0
  fi
  echo "$SAFETY_BANNER"
  read -r -p "Pilot ready and area clear? Type 'yes' to continue: " reply
  [[ "$reply" == "yes" ]] || die "Aborted."
}

do_preflight() {
  need_py
  echo "=== Preflight checklist ==="
  "$PY" scripts/capture_tello.py --preflight --min-battery "$MIN_BATTERY"
}

archive_capture() {
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local dest="$RAW/$stamp"
  mkdir -p "$dest"
  # Archive newest files from FLIGHTS (last 5 min) into dated raw folder.
  find "$FLIGHTS" -maxdepth 1 -type f \( -name '*.mp4' -o -name '*.csv' \) -mmin -5 \
    -exec cp -p {} "$dest/" \; 2>/dev/null || true
}

do_capture() {
  need_py
  mkdir -p "$RAW" "$FLIGHTS"
  local extra=(--out-dir "$FLIGHTS" --min-battery "$MIN_BATTERY")
  if [[ "$YES" == "1" ]]; then
    extra+=(--yes)
  fi
  "$PY" scripts/capture_tello.py "${extra[@]}" "$@"
  archive_capture
}

do_sessions() {
  need_py
  confirm_safety
  mkdir -p "$RAW" "$FLIGHTS"
  local count=0
  local entry prefix instruction

  echo ""
  echo "=== Maneuver session bundle (${DURATION}s each) ==="
  echo "Four tagged clips: hover, forward, turn, altitude."
  echo ""

  for entry in "${MANEUVERS[@]}"; do
    prefix="${entry%%|*}"
    instruction="${entry#*|}"
    echo "----------------------------------------"
    echo "Maneuver: $prefix"
    echo "  $instruction"
    if [[ "$YES" != "1" ]]; then
      read -r -p "  Ready to record? [Enter=record, s=skip]: " ans
      if [[ "$ans" == "s" || "$ans" == "S" ]]; then
        echo "  Skipped."
        continue
      fi
    fi
    do_capture \
      --duration "$DURATION" \
      --fps 15 \
      --name-prefix "$prefix" \
      --tags maneuver tello personal
    count=$((count + 1))
    if [[ "$YES" != "1" && "$prefix" != "altitude" ]]; then
      read -r -p "  Reposition for next maneuver. [Enter when ready]: " _
    fi
  done

  echo ""
  echo "Recorded $count maneuver clip(s) in $FLIGHTS"
  local n_mp4
  n_mp4=$(find "$FLIGHTS" -maxdepth 1 -name '*.mp4' | wc -l | tr -d ' ')
  [[ "$n_mp4" -gt 0 ]] || die "No captures saved. Check Tello link and retry."
}

do_preprocess() {
  need_py
  echo "=== Preprocess -> 128px / 15fps / 60s cap ==="
  if ! compgen -G "$FLIGHTS/*.mp4" > /dev/null && ! compgen -G "$FLIGHTS/*.MP4" > /dev/null; then
    die "No clips in $FLIGHTS. Run: $0 sessions --duration 30"
  fi
  "$PY" scripts/preprocess_real.py \
    --input-dir "$FLIGHTS" \
    --output-dir "$PROCESSED" \
    --target-fps 15 \
    --max-seconds 60 \
    --square \
    --resize 128 \
    --prefix tello
  echo ""
  "$PY" scripts/preprocess_real.py --probe --input-dir "$PROCESSED"
  echo ""
  echo "Training-ready clips: $PROCESSED"
}

do_train() {
  need_py
  need_baseline
  echo "=== Fine-tune from Wilds-adapted checkpoint ==="
  echo "  init:   $BASELINE_CKPT"
  echo "  config: $TELLO_CONFIG"
  echo "  data:   $PROCESSED"
  if ! compgen -G "$PROCESSED/*.mp4" > /dev/null; then
    die "No processed clips in $PROCESSED. Run: $0 preprocess"
  fi
  "$PY" scripts/train.py \
    --config "$TELLO_CONFIG" \
    --device "$DEVICE" \
    --init-checkpoint "$BASELINE_CKPT"
  [[ -f "$TELLO_CKPT" ]] || die "Training finished but $TELLO_CKPT not found."
  echo ""
  echo "Checkpoint saved: $TELLO_CKPT"
}

do_report() {
  need_py
  need_baseline
  [[ -f "$TELLO_CKPT" ]] || die "Missing $TELLO_CKPT — run: $0 train"
  echo "=== Transfer report: Wilds-only vs Tello fine-tuned ==="
  local wilds_flag=()
  if [[ ! -d "$WILDS" ]] || ! compgen -G "$WILDS/*.mp4" > /dev/null; then
    wilds_flag=(--skip-wilds-holdout)
    echo "(Wilds holdout skipped — $WILDS not found)"
  fi
  "$PY" scripts/tello_compare_report.py \
    --baseline-checkpoint "$BASELINE_CKPT" \
    --tello-checkpoint "$TELLO_CKPT" \
    --tello-data "$PROCESSED" \
    --wilds-data "$WILDS" \
    --device "$DEVICE" \
    "${wilds_flag[@]}"
  echo ""
  echo "Open results/tello_transfer_report.md for the summary."
}

do_all() {
  parse_duration "$@"
  parse_yes_flag "$@"
  do_preflight
  do_sessions "$@"
  do_preprocess
  do_train
  do_report
  echo ""
  echo "=== Done ==="
  echo "  Raw archive:     $RAW"
  echo "  Captures:        $FLIGHTS"
  echo "  Processed:       $PROCESSED"
  echo "  Checkpoint:      $TELLO_CKPT"
  echo "  Report:          results/tello_transfer_report.md"
}

print_help() {
  cat <<EOF
AeroJEPA Tello workflow — personal footage end-to-end

ONE COMMAND (recommended):
  $0 all --duration 30              # preflight + 4 maneuvers + train + report
  $0 all --duration 30 --yes          # skip interactive safety prompts

INDIVIDUAL STEPS:
  $0 preflight [--min-battery 20]     Wi-Fi, deps, battery checklist
  $0 capture --duration 30 --name-prefix hover --tags outdoor
  $0 sessions [--duration 30]         hover, forward, turn, altitude bundle
  $0 preprocess                       -> $PROCESSED
  $0 train                            fine-tune from $BASELINE_CKPT
  $0 report                           compare vs Wilds-only

FOLDERS:
  $RAW          archived raw captures (dated subfolders)
  $FLIGHTS      working captures (.mp4 + .csv + .raw.csv)
  $PROCESSED    128px training clips
  $TELLO_CKPT   output after train

TELEMETRY:
  .raw.csv  full Tello log (t, vgx, vgy, vgz, height, yaw, pitch, roll, ...)
  .csv      ACTION_COLUMNS deltas for training (derived automatically)

SAFETY:
  Record-only — never commands flight. See data/README.md for full notes.
EOF
}

cmd="${1:-help}"
shift || true

case "$cmd" in
  preflight)
    parse_duration "$@"
    do_preflight
    ;;
  capture)
    parse_yes_flag "$@"
    confirm_safety
    do_capture "$@"
    ;;
  sessions)
    parse_duration "$@"
    parse_yes_flag "$@"
    do_sessions
    ;;
  preprocess)
    do_preprocess
    ;;
  train)
    do_train
    ;;
  report|eval)
    do_report
    ;;
  all)
    do_all "$@"
    ;;
  help|-h|--help)
    print_help
    ;;
  *)
    die "Unknown command: $cmd (try: $0 help)"
    ;;
esac
