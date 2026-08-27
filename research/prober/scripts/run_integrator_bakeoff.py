#!/usr/bin/env python
"""SO(3) integrator bake-off (Prompt 7).

Same constant body rate ω; swap only the attitude update:
  Euclidean Euler on R^9 | Exp | RK4+project | LGVI midpoint.

Horizons 0.2 / 1 / 5 / 10 s. Zero-thrust free-fall for energy diagnostic.
Report ‖RᵀR−I‖_F, geodesic from init, energy drift.
Claim **constraint preservation**, never energy conservation.

Example::

    python research/prober/scripts/run_integrator_bakeoff.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "research" / "prober" / "src"))

from aerojepa_research.prober.integrator import _euler_ypr_to_rotation
from aerojepa_research.prober.metrics import geodesic_attitude_error_deg
from aerojepa_research.prober.so3_integrators import (
    EuclideanEulerR9,
    ExpIntegrator,
    LGVIMidpoint,
    RK4So3Integrator,
    _rotation_to_euler_ypr_deg,
    rotation_constraint_frobenius,
)


def _integrate_R(update_fn, R0: torch.Tensor, omega_rad: torch.Tensor, dt: float, steps: int):
    R = R0.clone()
    constr = []
    att = []
    for _ in range(steps):
        R = update_fn(R, omega_rad, dt)
        constr.append(rotation_constraint_frobenius(R))
        att.append(_rotation_to_euler_ypr_deg(R))
    return R, torch.stack(constr), torch.stack(att, dim=0)


def run_attitude_coast(name: str, update_fn, horizon_s: float, dt: float, omega_deg: torch.Tensor):
    steps = int(round(horizon_s / dt))
    R0 = _euler_ypr_to_rotation(torch.tensor([[15.0, -10.0, 5.0]]))
    omega = torch.deg2rad(omega_deg)
    R_final, constr, att = _integrate_R(update_fn, R0, omega, dt, steps)
    att0 = _rotation_to_euler_ypr_deg(R0).unsqueeze(0).expand(steps, -1, -1)
    # att: (T, 1, 3); geodesic wants matching shapes
    geo = geodesic_attitude_error_deg(att, att0)
    return {
        "name": name,
        "horizon_s": horizon_s,
        "steps": steps,
        "constraint_frobenius_mean": float(constr.mean()),
        "constraint_frobenius_final": float(constr[-1].mean()),
        "constraint_frobenius_max": float(constr.max()),
        "geodesic_from_init_final_deg": float(geo[-1].mean()),
        "constraint_curve": constr.squeeze(-1).tolist()[:: max(1, steps // 50)],
    }


def run_energy_freefall(dt: float, horizon_s: float):
    """Point-mass freefall under gravity (no attitude) — energy is *not* conserved
    under explicit Euler on v,p; included only as a dissipation diagnostic caption."""
    steps = int(round(horizon_s / dt))
    g = -9.81
    z = 0.0
    v = 0.0
    E0 = 0.0  # z=0 reference
    for _ in range(steps):
        v = v + g * dt
        z = z + v * dt
    E1 = 0.5 * v * v + (-g) * z  # KE + PE with PE = -g*z if g_vec=-9.81 on +z... 
    # With g_acc=-9.81 on +z: PE = m*|g|*z is wrong; use PE = -m*g_acc*z = m*9.81*z
    pe = 9.81 * z
    ke = 0.5 * v * v
    return {"energy_end_proxy": ke + pe, "z_final": z, "note": "explicit Euler freefall drifts"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=0.025)
    parser.add_argument("--horizons", default="0.2,1,5,10")
    parser.add_argument(
        "--out-dir",
        default=str(_ROOT / "research/prober/results/integrator_bakeoff"),
    )
    args = parser.parse_args()

    horizons = [float(x) for x in args.horizons.split(",")]
    omega_deg = torch.tensor([120.0, -80.0, 40.0])  # deg/s — persistently exciting
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Bind update methods without constructing full plant.
    makers = {
        "euclidean_euler_R9": EuclideanEulerR9(dt=args.dt)._update_attitude,
        "exp_first_order": ExpIntegrator(dt=args.dt)._update_attitude,
        "rk4_so3": RK4So3Integrator(dt=args.dt)._update_attitude,
        "lgvi_midpoint": LGVIMidpoint(dt=args.dt)._update_attitude,
    }

    report = {
        "dt": args.dt,
        "omega0_deg_s": omega_deg.tolist(),
        "claim": "constraint preservation (not energy conservation)",
        "citations": [
            "Marsden & West, Acta Numerica 2001 (variational integrators)",
            "Lee–Leok–McClamroch, CDC 2010 (geometric tracking on SE(3))",
            "SkyJEPA first-order Exp: Rao et al. arXiv:2606.23444",
        ],
        "integrators": {},
        "freefall_energy_note": run_energy_freefall(args.dt, horizons[-1]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, fn in makers.items():
        print(f"=== {name} ===")
        rows = []
        for h in horizons:
            row = run_attitude_coast(name, fn, h, args.dt, omega_deg)
            rows.append(row)
            print(
                f"  H={h:5.1f}s  ‖RᵀR−I‖_F final={row['constraint_frobenius_final']:.3e}  "
                f"max={row['constraint_frobenius_max']:.3e}  "
                f"geo={row['geodesic_from_init_final_deg']:.1f}°"
            )
        report["integrators"][name] = rows
        long = rows[-1]
        t = np.linspace(0, long["horizon_s"], len(long["constraint_curve"]))
        axes[0].semilogy(t, np.maximum(long["constraint_curve"], 1e-16), label=name)
        axes[1].plot(
            [r["horizon_s"] for r in rows],
            [np.maximum(r["constraint_frobenius_final"], 1e-16) for r in rows],
            marker="o",
            label=name,
        )

    axes[0].set_xlabel("t (s) constant-ω coast")
    axes[0].set_ylabel("‖RᵀR − I‖_F")
    axes[0].set_title(f"Constraint drift @ H={horizons[-1]} s")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("horizon (s)")
    axes[1].set_ylabel("final ‖RᵀR − I‖_F")
    axes[1].set_yscale("log")
    axes[1].set_title("Constraint vs horizon")
    axes[1].legend(fontsize=7)
    fig.suptitle("Integrator bake-off — constraint preservation (not energy)", fontsize=10)
    fig.tight_layout()
    fig_path = out_dir / "integrator_bakeoff.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    out_json = out_dir / "integrator_bakeoff.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
