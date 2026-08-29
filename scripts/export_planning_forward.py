#!/usr/bin/env python
"""Export frozen planning encoder+predictor (TorchScript + optional ONNX).

CPU-first. Does not export residual heads or PyFlyt. Same 3-5M world model.

Example::

    python scripts/export_planning_forward.py \\
        --checkpoint checkpoints/action_conditioned_wilds/latest.pt \\
        --out-dir exports
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import torch

from aerojepa.eval import load_model
from aerojepa.export.planning_forward import build_planning_forward, planning_shapes
from aerojepa.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/action_conditioned_wilds/latest.pt",
    )
    parser.add_argument("--out-dir", default="exports")
    parser.add_argument("--context-frames", type=int, default=None)
    parser.add_argument("--batch", type=int, default=1, help="Dummy batch for trace/export.")
    parser.add_argument("--onnx", action="store_true", help="Also write ONNX (opset 17).")
    parser.add_argument("--device", default="cpu", help="Export device (cpu first).")
    args = parser.parse_args()

    device = get_device(args.device)
    if device.type != "cpu":
        print(f"note: exporting on {device}; CPU is the default target")

    model, cfg = load_model(args.checkpoint, device)
    img_size = int(cfg["data"]["img_size"])
    pf = build_planning_forward(model, context_frames=args.context_frames).to(device)
    shapes = planning_shapes(
        model, context_frames=pf.context_frames, img_size=img_size
    )
    print(
        f"PlanningForward params={shapes['param_count']:,} "
        f"C={shapes['context_frames']} H={shapes['horizon']} "
        f"img={img_size} action_cond={shapes['action_conditioned']}"
    )
    # Keep the 3-5M class model; do not grow layers here.
    if shapes["param_count"] > 6_000_000:
        raise SystemExit(
            f"param_count {shapes['param_count']} exceeds ~5M class; refusing export"
        )

    b = int(args.batch)
    context = torch.rand(b, pf.context_frames, 3, img_size, img_size, device=device)
    actions = torch.zeros(b, pf.num_temporal, 6, device=device)
    with torch.no_grad():
        out = pf(context, actions)
    print(f"eager out shape={tuple(out.shape)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "planning_forward_meta.json"
    meta_path.write_text(json.dumps({**shapes, "checkpoint": args.checkpoint}, indent=2) + "\n")

    # TorchScript (trace): CPU-friendly frozen graph.
    pf_cpu = pf.to("cpu").eval()
    context_cpu = context.detach().cpu()
    actions_cpu = actions.detach().cpu()
    with torch.no_grad():
        traced = torch.jit.trace(pf_cpu, (context_cpu, actions_cpu), strict=False)
    ts_path = out_dir / "planning_forward.ts"
    traced.save(str(ts_path))
    print(f"wrote {ts_path}")

    if args.onnx:
        onnx_path = out_dir / "planning_forward.onnx"
        torch.onnx.export(
            pf_cpu,
            (context_cpu, actions_cpu),
            str(onnx_path),
            input_names=["context_clip", "actions_full"],
            output_names=["pred_latents"],
            dynamic_axes={
                "context_clip": {0: "batch"},
                "actions_full": {0: "batch"},
                "pred_latents": {0: "batch"},
            },
            opset_version=17,
        )
        print(f"wrote {onnx_path}")

    print(f"meta -> {meta_path}")


if __name__ == "__main__":
    main()
