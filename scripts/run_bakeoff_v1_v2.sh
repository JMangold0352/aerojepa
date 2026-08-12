#!/usr/bin/env bash
# Bake-off: Wilds action closed-loop stack v1 (default) vs v2 (continuation).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl-aerojepa}"
PY="${PY:-$ROOT/.venv/bin/python}"
LOG="$ROOT/logs/bakeoff_v1_v2.log"
OUTDIR="$ROOT/visualizations/closed_loop"
mkdir -p "$OUTDIR" "$ROOT/logs"

exec > >(tee -a "$LOG") 2>&1
echo "=== BAKEOFF START $(date -Iseconds) ==="

echo "=== v1 (default) ==="
"$PY" scripts/compare_full_stack.py \
  --checkpoint checkpoints/action_conditioned_wilds/latest.pt \
  --residual checkpoints/action_residual_wilds/best.pt \
  --seeds 0 1 2 \
  --latent-smooth 0.05 \
  --include-hard-turn \
  --out "$OUTDIR/full_stack_compare_wilds_v1_bakeoff.json"

echo "=== v2 ==="
"$PY" scripts/compare_full_stack.py \
  --checkpoint checkpoints/action_conditioned_wilds_v2/latest.pt \
  --residual checkpoints/action_residual_wilds_v2/best.pt \
  --seeds 0 1 2 \
  --latent-smooth 0.05 \
  --include-hard-turn \
  --out "$OUTDIR/full_stack_compare_wilds_v2_bakeoff.json"

"$PY" - <<'PY'
import json
from pathlib import Path

def load(p):
    return json.loads(Path(p).read_text())

v1 = load("visualizations/closed_loop/full_stack_compare_wilds_v1_bakeoff.json")
v2 = load("visualizations/closed_loop/full_stack_compare_wilds_v2_bakeoff.json")
rows = []
for task in v1["tasks"]:
    a1 = v1["tasks"][task]["aggregate"]["full_stack"]
    a2 = v2["tasks"][task]["aggregate"]["full_stack"]
    rows.append({
        "task": task,
        "v1_success": a1["success_rate"],
        "v2_success": a2["success_rate"],
        "v1_max_xy": a1["mean_max_xy_drift"],
        "v2_max_xy": a2["mean_max_xy_drift"],
    })
out = {
    "v1": {
        "world": v1["world_checkpoint"],
        "residual": v1["residual_checkpoint"],
        "protocol_b_real_cosine": 0.957,
    },
    "v2": {
        "world": v2["world_checkpoint"],
        "residual": v2["residual_checkpoint"],
        "protocol_b_real_cosine": 0.9145,
    },
    "seeds": v1["seeds"],
    "tasks": rows,
}
Path("visualizations/closed_loop/bakeoff_v1_v2_summary.json").write_text(
    json.dumps(out, indent=2) + "\n"
)
print("\n=== BAKEOFF SUMMARY ===")
print(f"{'task':<22} {'v1 ok':>7} {'v2 ok':>7} {'v1 max_xy':>10} {'v2 max_xy':>10}")
for r in rows:
    print(
        f"{r['task']:<22} {r['v1_success']:>7.0%} {r['v2_success']:>7.0%} "
        f"{r['v1_max_xy']:>10.3f} {r['v2_max_xy']:>10.3f}"
    )
print("→ visualizations/closed_loop/bakeoff_v1_v2_summary.json")
PY

echo "=== BAKEOFF_DONE $(date -Iseconds) ==="
