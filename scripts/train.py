#!/usr/bin/env python
"""Train an AeroJEPA model from a YAML config.

Examples::

    python scripts/train.py --config configs/aerojepa_baseline.yaml
    python scripts/train.py --config configs/aerojepa_world_model.yaml --device mps

    # Fine-tune a synthetic world model on real footage:
    python scripts/train.py --config configs/aerojepa_real.yaml \
        --init-checkpoint checkpoints/world_model/latest.pt

    # Resume an interrupted run (restores optimizer + scheduler + epoch):
    python scripts/train.py --config configs/aerojepa_finetune.yaml \
        --resume checkpoints/real_finetune/latest.pt
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from aerojepa.train import train
from aerojepa.utils.config import load_config
from aerojepa.utils.device import get_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a training config YAML.")
    parser.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Warm-start weights from this checkpoint (fine-tuning). "
        "Overrides train.init_checkpoint in the config. Ignored when --resume is set.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="CHECKPOINT",
        help="Resume a prior run from latest.pt (model + optimizer + scheduler + epoch). "
        "Overrides train.resume_checkpoint in the config.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(args.device)
    print(f"Device: {device}")
    checkpoint = train(
        cfg,
        device,
        init_checkpoint=args.init_checkpoint,
        resume_checkpoint=args.resume,
    )
    print(f"Done. Latest checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
