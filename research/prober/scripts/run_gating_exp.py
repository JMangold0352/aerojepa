#!/usr/bin/env python
"""Gating + body residual experiment.

Frozen JEPA encoder. Three prober variants + SO(3) frame invariance test.
Results → research/prober/results/gating_exp/ + updates gating_exp.md narrative.

Example::

    python research/prober/scripts/run_gating_exp.py --epochs 5 --num-train 64 --skip-train
    python research/prober/scripts/run_gating_exp.py --epochs 5 --num-train 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT / "research" / "prober" / "src"))
sys.path.insert(0, str(_ROOT / "src"))

from aerojepa_research.prober.data_pyflyt import build_pyflyt_dataloaders
from aerojepa_research.prober.gating import (
    GatedBodyProber,
    PartialGateProber,
    UngatedWorldProber,
    frame_invariance_test,
)
from aerojepa_research.prober.integrator import (
    ControlIntegrator,
    MetricState,
    _euler_ypr_to_rotation,
    wrap_degrees,
)
from aerojepa_research.prober.loss import metric_state_mse
from aerojepa_research.prober.metrics import compute_metrics
from aerojepa_research.prober.rollout import FrozenRolloutExtractor
from aerojepa_research.prober.so3_integrators import so3_exp


def _att_from_nominal(integ, init: MetricState, controls: torch.Tensor) -> torch.Tensor:
    """Open-loop attitude estimate from nominal rates (no future GT). (B,T,3)."""
    st = init.clone()
    att_list = []
    for t in range(controls.shape[1]):
        att_list.append(st.euler_att)
        _, a_ang_n = integ.nominal_accel(controls[:, t], st)
        new_av = st.ang_vel + a_ang_n * integ.dt
        new_att = wrap_degrees(st.euler_att + new_av * integ.dt)
        st = MetricState(pos=st.pos, vel=st.vel, euler_att=new_att, ang_vel=new_av)
    return torch.stack(att_list, dim=1)


def _residuals(name, module, latents, controls, att):
    if name == "A_ungated_world":
        return module(latents, controls)
    return module(latents, controls, att)


def _train_variant(name, module, extractor, loader, device, epochs, lr, integ):
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    module.train()
    hist = []
    for ep in range(epochs):
        total, n = 0.0, 0
        for clips, _actions, states, controls in loader:
            clips = clips.to(device)
            controls = controls.to(device)
            states = states.to(device)
            roll = extractor.extract(clips, controls, states)
            att = _att_from_nominal(integ, roll.init_state, roll.controls)
            res_lin, res_ang = _residuals(name, module, roll.latents, roll.controls, att)
            pred = integ.rollout(roll.init_state, roll.controls, res_lin, res_ang)
            pred_stack = torch.cat(
                [pred.pos, pred.vel, pred.euler_att, pred.ang_vel], dim=-1
            )
            loss = metric_state_mse(pred_stack, roll.gt_states)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        hist.append(total / max(n, 1))
        print(f"  {name} ep {ep + 1}/{epochs} loss={hist[-1]:.5f}")
    return hist


@torch.no_grad()
def _eval_variant(name, module, extractor, loader, device, integ):
    module.eval()
    preds, gts = [], []
    for clips, _actions, states, controls in loader:
        clips = clips.to(device)
        controls = controls.to(device)
        states = states.to(device)
        roll = extractor.extract(clips, controls, states)
        att = _att_from_nominal(integ, roll.init_state, roll.controls)
        res_lin, res_ang = _residuals(name, module, roll.latents, roll.controls, att)
        pred = integ.rollout(roll.init_state, roll.controls, res_lin, res_ang)
        pred_stack = torch.cat(
            [pred.pos, pred.vel, pred.euler_att, pred.ang_vel], dim=-1
        )
        preds.append(pred_stack.cpu())
        gts.append(roll.gt_states.cpu())
    return compute_metrics(torch.cat(preds), torch.cat(gts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(_ROOT / "research/prober/configs/prober_synth.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--num-train", type=int, default=64)
    parser.add_argument("--num-val", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-dir",
        default=str(_ROOT / "research/prober/results/gating_exp"),
    )
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    a_body = torch.randn(64, 3)
    R0 = _euler_ypr_to_rotation(torch.tensor([[10.0, -5.0, 20.0]]))[0]
    a_world = (R0 @ a_body.T).T
    R_extra = so3_exp(torch.tensor([0.3, -0.2, 0.5]))
    inv = frame_invariance_test(a_body, a_world, R_extra)
    print("frame invariance:", inv)

    report: dict = {
        "skyjepa_note": (
            "SkyJEPA (Rao et al., arXiv:2606.23444) uses a world-frame Δv̇ residual. "
            "This body-vs-world SO(3) invariance test is not in their paper."
        ),
        "frame_invariance": inv,
        "variants": {},
        "prediction": (
            "Gated body residual should win at 1-5 s with a t^2 signature, "
            "not necessarily at 0.1 s (4 frames)."
        ),
    }

    if not args.skip_train:
        ckpt = Path(cfg["checkpoint"])
        if not ckpt.is_file():
            ckpt = _ROOT / cfg["checkpoint"]
        extractor = FrozenRolloutExtractor(ckpt, device=device)
        train_loader, val_loader = build_pyflyt_dataloaders(
            batch_size=args.batch_size,
            num_frames=cfg["data"]["num_frames"],
            img_size=cfg["data"]["img_size"],
            num_train=args.num_train,
            num_val=args.num_val,
            num_workers=0,
            seed=args.seed,
        )
        integ = ControlIntegrator(
            dt=cfg["integrator"]["dt"],
            gravity=cfg["integrator"]["gravity"],
            mass=cfg["integrator"].get("mass", 1.0),
            hover_thrust=float(cfg["integrator"].get("hover_thrust", 0.39)),
        ).to(device)

        variants = {
            "A_ungated_world": UngatedWorldProber(
                hidden_dim=cfg["prober"]["hidden_dim"],
                num_layers=cfg["prober"]["num_layers"],
            ).to(device),
            "B_gated_body": GatedBodyProber(
                hidden_dim=cfg["prober"]["hidden_dim"],
                num_layers=cfg["prober"]["num_layers"],
            ).to(device),
            "C_partial_rz": PartialGateProber(
                hidden_dim=cfg["prober"]["hidden_dim"],
                num_layers=cfg["prober"]["num_layers"],
            ).to(device),
        }
        for name, mod in variants.items():
            print(f"=== train {name} ===")
            hist = _train_variant(
                name,
                mod,
                extractor,
                train_loader,
                device,
                args.epochs,
                float(cfg["train"].get("lr", 1e-3)),
                integ,
            )
            m = _eval_variant(name, mod, extractor, val_loader, device, integ)
            report["variants"][name] = {
                "train_loss": hist,
                "position_rmse": m.position_rmse,
                "attitude_rmse_geodesic_deg": m.attitude_rmse_geodesic_deg,
                "attitude_rmse_deg_legacy": m.attitude_rmse_deg,
                "per_horizon_pos_rmse": m.per_horizon_pos_rmse,
                "per_horizon_att_geodesic_rmse": m.per_horizon_att_geodesic_rmse,
            }
            print(
                f"  eval pos={m.position_rmse:.4f} m  "
                f"att_geo={m.attitude_rmse_geodesic_deg:.3f}°"
            )

    (out_dir / "gating_results.json").write_text(json.dumps(report, indent=2))
    print(f"Wrote {out_dir / 'gating_results.json'}")


if __name__ == "__main__":
    main()
