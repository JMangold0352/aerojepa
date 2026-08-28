#!/usr/bin/env bash
# Download released pretrained checkpoints into checkpoints/ (idempotent).
#
# Usage:
#   ./scripts/download_weights.sh --list
#   ./scripts/download_weights.sh
#   ./scripts/download_weights.sh world_model
#
# URLs: released_weights/urls.yaml or AEROJEPA_WEIGHT_URL_* env vars.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d ".venv/bin" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

exec python scripts/download_weights.py "$@"
