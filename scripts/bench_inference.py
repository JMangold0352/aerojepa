#!/usr/bin/env python
"""Bench planning-forward latency (p50/p95) on CPU and MPS.

Matches closed-loop defaults: img_size=64, context=T//2, horizon=T-context,
batch = num_candidates (default 12, gradient planner). Writes
``results/inference_latency.json``.

If p95 > 25 ms (40 Hz budget), prefer a larger ``--replan-every`` or a shorter
``--horizon`` in the closed-loop CLI rather than growing the ~3-5M world model.

Example::

    python scripts/bench_inference.py \\
        --checkpoint checkpoints/action_conditioned_wilds/latest.pt
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import torch

from aerojepa.eval import load_model
from aerojepa.export.planning_forward import build_planning_forward, planning_shapes
from aerojepa.utils.device import get_device

# Closed-loop wall budget at agent_hz=40.
BUDGET_MS = 25.0


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(ys) - 1)
    if f == c:
        return float(ys[f])
    return float(ys[f] + (ys[c] - ys[f]) * (k - f))


def _round_stats(stats: dict[str, float]) -> dict[str, float | int]:
    return {
        "p50_ms": round(float(stats["p50_ms"]), 2),
        "p95_ms": round(float(stats["p95_ms"]), 2),
        "mean_ms": round(float(stats["mean_ms"]), 2),
        "n": int(stats["n"]),
    }


def _bench_module(
    module: torch.nn.Module,
    context: torch.Tensor,
    actions: torch.Tensor,
    *,
    warmup: int,
    runs: int,
) -> dict[str, float]:
    module.eval()
    with torch.no_grad():
        for _ in range(warmup):
            _ = module(context, actions)
        if context.device.type == "mps":
            torch.mps.synchronize()
        times: list[float] = []
        for _ in range(runs):
            if context.device.type == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = module(context, actions)
            if context.device.type == "mps":
                torch.mps.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return {
        "p50_ms": _percentile(times, 50),
        "p95_ms": _percentile(times, 95),
        "mean_ms": float(statistics.fmean(times)),
        "n": float(runs),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/action_conditioned_wilds/latest.pt",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=12,
        help="Candidate batch size (gradient planner default).",
    )
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--out", default="results/inference_latency.json")
    parser.add_argument(
        "--torchscript",
        default="exports/planning_forward.ts",
        help="Optional traced module to bench alongside eager (CPU only).",
    )
    args = parser.parse_args()

    model_cpu, cfg = load_model(args.checkpoint, torch.device("cpu"))
    img_size = int(cfg["data"]["img_size"])
    shapes = planning_shapes(model_cpu, context_frames=args.context_frames, img_size=img_size)
    print(
        f"shapes: C={shapes['context_frames']} H={shapes['horizon']} "
        f"img={img_size} params={shapes['param_count']:,} batch={args.batch}"
    )

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    results: dict = {
        "protocol": {
            "forward": "encoder+predictor (PlanningForward); residual and sim excluded",
            "shapes": "closed-loop defaults (context=T//2, horizon=T-context)",
            "batch": "num_candidates for gradient planner (default 12)",
            "budget_ms": BUDGET_MS,
            "agent_hz": 40,
        },
        "checkpoint": args.checkpoint,
        "budget_ms": BUDGET_MS,
        "batch": args.batch,
        "img_size": img_size,
        "context_frames": shapes["context_frames"],
        "horizon": shapes["horizon"],
        "param_count": shapes["param_count"],
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "note": (
            "If p95_ms > budget_ms, increase --replan-every or shorten --horizon "
            "in closed-loop / shadow CLIs; do not grow the world-model depth/width."
        ),
        "devices": {},
    }

    ts_path = Path(args.torchscript)
    traced = None
    if ts_path.exists():
        traced = torch.jit.load(str(ts_path), map_location="cpu")
        traced.eval()
        print(f"loaded TorchScript {ts_path}")

    for dev_name in devices:
        device = get_device(dev_name)
        model, _ = load_model(args.checkpoint, device)
        pf = build_planning_forward(model, context_frames=args.context_frames).to(device)
        context = torch.rand(
            args.batch,
            pf.context_frames,
            3,
            img_size,
            img_size,
            device=device,
        )
        actions = torch.zeros(args.batch, pf.num_temporal, 6, device=device)
        eager = _round_stats(
            _bench_module(pf, context, actions, warmup=args.warmup, runs=args.runs)
        )
        entry: dict = {
            "eager": eager,
            "over_budget": eager["p95_ms"] > BUDGET_MS,
        }
        if traced is not None and device.type == "cpu":
            entry["torchscript"] = _round_stats(
                _bench_module(
                    traced,
                    context.cpu(),
                    actions.cpu(),
                    warmup=args.warmup,
                    runs=args.runs,
                )
            )
        results["devices"][dev_name] = entry
        flag = " (over budget)" if entry["over_budget"] else ""
        print(
            f"  {dev_name}: p50={eager['p50_ms']:.2f} ms  "
            f"p95={eager['p95_ms']:.2f} ms{flag}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
