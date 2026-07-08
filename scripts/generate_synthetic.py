#!/usr/bin/env python
"""Render a few synthetic drone clips to disk for inspection.

Saves a contact sheet (frames laid out in a grid) and the telemetry so you can
see what the model actually trains on. Purely a visualization aid -- training
generates clips on the fly and needs nothing on disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

import matplotlib.pyplot as plt

from aerojepa.data.synthetic import render_clip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-clips", type=int, default=3)
    parser.add_argument("--num-frames", type=int, default=8)
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--out-dir", default="results/synthetic_preview")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for clip_idx in range(args.num_clips):
        clip = render_clip(seed=clip_idx, num_frames=args.num_frames, img_size=args.img_size)
        fig, axes = plt.subplots(1, args.num_frames, figsize=(2 * args.num_frames, 2.4))
        for t, ax in enumerate(axes):
            ax.imshow(clip.frames[t].permute(1, 2, 0).clamp(0, 1).numpy())
            ax.set_title(f"t={t}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Synthetic drone clip #{clip_idx} (moving 6-DoF camera)", fontsize=11)
        fig.tight_layout()
        path = out_dir / f"clip_{clip_idx:02d}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {path}")

    print(f"Done. Previews in {out_dir}")


if __name__ == "__main__":
    main()
