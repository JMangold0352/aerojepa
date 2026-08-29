# Planning forward export

Export the frozen **encoder + predictor** used by `LatentPlanner` (pad context,
encode, predict). Residual heads and the simulator are not part of this graph.

Default shapes match closed-loop: `img_size=64`, `context_frames = T // 2`,
`horizon = T - context` (4 / 4 when `num_frames=8`). The exported module stays
in the ~3–5M parameter class.

## Export (CPU first)

```bash
python scripts/export_planning_forward.py \
  --checkpoint checkpoints/action_conditioned_wilds/latest.pt \
  --out-dir exports
# optional: --onnx
```

Produces:

| Artifact | Notes |
| --- | --- |
| `exports/planning_forward.ts` | TorchScript (trace); gitignored |
| `exports/planning_forward_meta.json` | Shapes + param count; gitignored |

Regenerate locally after changing the checkpoint. Optional ONNX uses opset 17.

## Latency bench

```bash
python scripts/bench_inference.py \
  --checkpoint checkpoints/action_conditioned_wilds/latest.pt
```

Measures p50 / p95 on CPU and MPS (when available) at the shapes above.
Default `--batch 12` matches the gradient planner candidate count. Writes
[`results/inference_latency.json`](../results/inference_latency.json).

Closed-loop steps target **25 ms** at `agent_hz=40`. When planning-forward p95
exceeds that budget, prefer a larger `--replan-every` or a shorter `--horizon`
in the closed-loop / shadow CLIs rather than growing the network. TensorRT /
Jetson Orin packaging is out of scope for this path.
