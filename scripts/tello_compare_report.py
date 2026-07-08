#!/usr/bin/env python
"""Compare Wilds-only vs Tello-fine-tuned checkpoints on personal footage.

Evaluates both checkpoints with ``evaluate_real_gap`` and writes a JSON report
plus a human-readable Markdown summary for README / portfolio use.

Examples::

    python scripts/tello_compare_report.py
    python scripts/tello_compare_report.py \
        --baseline-checkpoint checkpoints/real_finetune_fast/latest.pt \
        --tello-checkpoint checkpoints/real_finetune_tello/latest.pt \
        --tello-data data/flights_tello_128
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import _bootstrap  # noqa: F401

from aerojepa.eval.real_gap import evaluate_real_gap


def _latent_cos(report: dict) -> float:
    return float(report["real"]["latent_prediction"]["cosine"])


def _rollout_at_4(report: dict) -> float | None:
    roll = report["real"]["rollout"]
    horizons = roll.get("horizon", [])
    cosines = roll.get("cosine", [])
    if 4 in horizons:
        return float(cosines[horizons.index(4)])
    return float(cosines[-1]) if cosines else None


def _gap(report: dict) -> float:
    return float(report["gap"]["latent_cosine"])


def build_markdown(payload: dict) -> str:
    t = payload["tello_data_eval"]
    w = payload.get("wilds_data_eval", {})
    b = t["baseline_wilds_only"]
    f = t["tello_finetuned"]
    delta_latent = f["real_latent_cosine"] - b["real_latent_cosine"]
    delta_gap = b["sim_to_real_gap"] - f["sim_to_real_gap"]

    lines = [
        "# Tello transfer report",
        "",
        f"Generated: {payload['generated_at']}",
        "",
        "## Summary",
        "",
        "Personal Tello footage vs the Wilds-only fine-tune baseline.",
        "",
        "| Checkpoint | Real latent cosine | Rollout @ 4 | Sim-to-real gap |",
        "| --- | ---: | ---: | ---: |",
        f"| Wilds-only (`real_finetune_fast`) | {b['real_latent_cosine']:.4f} | "
        f"{b['rollout_cosine_at_4']:.4f} | {b['sim_to_real_gap']:+.4f} |",
        f"| Wilds + Tello (`real_finetune_tello`) | {f['real_latent_cosine']:.4f} | "
        f"{f['rollout_cosine_at_4']:.4f} | {f['sim_to_real_gap']:+.4f} |",
        "",
        f"**Tello real cosine delta:** {delta_latent:+.4f} "
        f"(positive = Tello fine-tune helped on your footage)",
        "",
        f"**Gap closure on Tello data:** {delta_gap:+.4f} "
        f"(positive = sim-to-real gap shrank after Tello fine-tune)",
        "",
    ]

    if w:
        bw = w["baseline_wilds_only"]
        fw = w["tello_finetuned"]
        lines.extend([
            "## Wilds holdout (forgetting check)",
            "",
            "| Checkpoint | Real latent cosine on Wilds |",
            "| --- | ---: |",
            f"| Wilds-only | {bw['real_latent_cosine']:.4f} |",
            f"| After Tello fine-tune | {fw['real_latent_cosine']:.4f} |",
            "",
        ])

    lines.extend([
        "## Data",
        "",
        f"- Tello eval folder: `{payload['tello_data_dir']}`",
        f"- Baseline checkpoint: `{payload['baseline_checkpoint']}`",
        f"- Tello checkpoint: `{payload['tello_checkpoint']}`",
        "",
        "## Telemetry",
        "",
        "Tello captures write `.raw.csv` (full log) and `.csv` (ACTION_COLUMNS deltas).",
        "Preprocess keeps sibling telemetry aligned; missing CSVs default actions to zero.",
        "",
    ])
    return "\n".join(lines)


def summarize_eval(label: str, report: dict) -> dict:
    r4 = _rollout_at_4(report)
    return {
        "label": label,
        "checkpoint": report["checkpoint"],
        "data_dir": report["data_dir"],
        "real_latent_cosine": _latent_cos(report),
        "rollout_cosine_at_4": r4 if r4 is not None else float("nan"),
        "sim_to_real_gap": _gap(report),
        "full": report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-checkpoint",
        default="checkpoints/real_finetune_fast/latest.pt",
        help="Wilds-only model before Tello fine-tune.",
    )
    parser.add_argument(
        "--tello-checkpoint",
        default="checkpoints/real_finetune_tello/latest.pt",
        help="Model after fine-tune on personal Tello footage.",
    )
    parser.add_argument("--tello-data", default="data/flights_tello_128")
    parser.add_argument("--wilds-data", default="data/flights_128")
    parser.add_argument(
        "--skip-wilds-holdout", action="store_true",
        help="Skip re-evaluating both models on Wilds clips.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-batches", type=int, default=8)
    parser.add_argument("--out-json", default="results/tello_transfer_report.json")
    parser.add_argument("--out-md", default="results/tello_transfer_report.md")
    args = parser.parse_args()

    tello_dir = Path(args.tello_data)
    if not tello_dir.is_dir() or not list(tello_dir.glob("*.mp4")):
        raise SystemExit(
            f"No Tello clips in {tello_dir}. Run ./scripts/tello_workflow.sh preprocess first."
        )

    for ckpt in (args.baseline_checkpoint, args.tello_checkpoint):
        if not Path(ckpt).is_file():
            raise SystemExit(f"Missing checkpoint: {ckpt}")

    kw = dict(device=args.device, max_batches=args.max_batches)

    print("=== baseline (Wilds-only) on Tello footage ===")
    base_tello = evaluate_real_gap(args.baseline_checkpoint, args.tello_data, **kw)
    print(f"real latent cosine={_latent_cos(base_tello):.4f}, gap={_gap(base_tello):+.4f}")

    print("=== Tello fine-tuned on Tello footage ===")
    ft_tello = evaluate_real_gap(args.tello_checkpoint, args.tello_data, **kw)
    print(f"real latent cosine={_latent_cos(ft_tello):.4f}, gap={_gap(ft_tello):+.4f}")

    payload: dict = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "baseline_checkpoint": args.baseline_checkpoint,
        "tello_checkpoint": args.tello_checkpoint,
        "tello_data_dir": str(tello_dir),
        "tello_data_eval": {
            "baseline_wilds_only": summarize_eval("wilds_only", base_tello),
            "tello_finetuned": summarize_eval("tello_finetuned", ft_tello),
            "delta": {
                "real_latent_cosine": _latent_cos(ft_tello) - _latent_cos(base_tello),
                "sim_to_real_gap": _gap(base_tello) - _gap(ft_tello),
            },
        },
    }

    wilds_dir = Path(args.wilds_data)
    if not args.skip_wilds_holdout and wilds_dir.is_dir() and list(wilds_dir.glob("*.mp4")):
        print("=== Wilds holdout (both checkpoints) ===")
        base_w = evaluate_real_gap(args.baseline_checkpoint, args.wilds_data, **kw)
        ft_w = evaluate_real_gap(args.tello_checkpoint, args.wilds_data, **kw)
        payload["wilds_data_dir"] = str(wilds_dir)
        payload["wilds_data_eval"] = {
            "baseline_wilds_only": summarize_eval("wilds_only", base_w),
            "tello_finetuned": summarize_eval("tello_finetuned", ft_w),
            "delta": {
                "real_latent_cosine": _latent_cos(ft_w) - _latent_cos(base_w),
            },
        }

    json_path = Path(args.out_json)
    md_path = Path(args.out_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2))
    md_path.write_text(build_markdown(payload))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
