from __future__ import annotations

import matplotlib as mpl

# A single, consistent look for every figure: publication DPI, readable fonts,
# and a restrained palette. Import ``apply_style()`` before plotting.

PALETTE = {
    "baseline": "#4C72B0",
    "looped": "#DD8452",
    "target": "#54A24B",
    "accent": "#C44E52",
    "muted": "#8C8C8C",
}

DPI = 300


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "legend.frameon": False,
            "image.cmap": "viridis",
        }
    )
