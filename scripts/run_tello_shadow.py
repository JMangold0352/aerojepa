#!/usr/bin/env python
"""Tello shadow observer: propose (vp,vq,vr,T) beside a human-piloted session.

Observer only. Never applies controls. Pair with ``capture_tello.py`` (separate
process / stream owner) or run offline against a finished capture.

Live::

    # Terminal A (record):
    python scripts/capture_tello.py --duration 30 --name-prefix flight --yes
    # Terminal B (shadow) - use the same session stem after capture finishes,
    # or run offline:
    python scripts/run_tello_shadow.py \\
        --checkpoint checkpoints/action_conditioned_wilds/latest.pt \\
        --capture-raw data/flights/flight_YYYYMMDD_HHMMSS.raw.csv \\
        --video data/flights/flight_YYYYMMDD_HHMMSS.mp4

Offline alignment writes ``<session>_shadow.jsonl`` with the same ``t`` column
as the capture raw CSV (nearest frame by time).

Live stream mode (this process owns the camera; do not also capture on the
same Wi-Fi client)::

    python scripts/run_tello_shadow.py --live --duration 30 --name-prefix shadow \\
        --checkpoint checkpoints/action_conditioned_wilds/latest.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np
import torch

from aerojepa.eval import load_model
from aerojepa.sim.action_residual import load_residual_head, map_aero_with_optional_residual
from aerojepa.sim.planner import LatentPlanner, MultiStepCostWeights
from aerojepa.sim.vehicle import DEFAULT_HOVER_THRUST, clip_control
from aerojepa.utils.device import get_device

SAFETY_BANNER = """
========================== TELLO SHADOW - OBSERVER ============================
 * Human flies. This process only reads RGB/telemetry and logs proposed controls.
 * It refuses Vehicle.step(); nothing here moves the aircraft.
 * Battery / preflight reuse capture_tello checks (--preflight / --min-battery).
 * Prefer offline --capture-raw + --video so logs share capture timestamps.
 * Live --live owns the stream; do not run capture on the same client at once.
===============================================================================
"""


def _session_stem(prefix: str, tags: list[str]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [prefix, stamp, *tags]
    return "_".join(p for p in parts if p)


def _load_raw_times(path: Path) -> np.ndarray:
    rows = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return rows[:, 0].astype(np.float64)


def _frame_at(video_path: Path, t_s: float, fps_hint: float) -> np.ndarray | None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint
    idx = int(max(0, round(t_s * float(fps))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _resize_chw01(rgb_u8: np.ndarray, img_size: int) -> np.ndarray:
    import cv2

    h, w = rgb_u8.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = rgb_u8[y0 : y0 + side, x0 : x0 + side]
    if crop.shape[0] != img_size:
        crop = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return np.transpose(crop.astype(np.float32) / 255.0, (2, 0, 1))


def _propose(
    planner: LatentPlanner,
    buffer: deque[np.ndarray],
    *,
    context_frames: int,
    residual_head,
    hover_thrust: float,
    num_candidates: int,
    action_scale: float,
    seed: int,
) -> tuple[np.ndarray, float, float]:
    """Return clipped (vp,vq,vr,T), plan_cost, encode_plan_ms."""
    t0 = time.perf_counter()
    while len(buffer) < context_frames:
        buffer.appendleft(buffer[0].copy())
    ctx = torch.from_numpy(np.stack(list(buffer)[-context_frames:]))
    result = planner.plan(
        ctx,
        num_candidates=num_candidates,
        action_scale=action_scale,
        seed=seed,
    )
    mapped = map_aero_with_optional_residual(
        result.best_actions[0].numpy(),
        residual_head=residual_head,
        latent=result.best_latents[0] if result.best_latents is not None else None,
        hover_thrust=hover_thrust,
    )
    control = clip_control(mapped)
    encode_ms = (time.perf_counter() - t0) * 1000.0
    cost = float(result.costs[result.best_index])
    return control, cost, encode_ms


def run_offline(args: argparse.Namespace) -> Path:
    device = get_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    img_size = int(cfg["data"]["img_size"])
    latent_dim = int(cfg["encoder"]["embed_dim"])
    residual = None
    if args.residual_checkpoint and Path(args.residual_checkpoint).exists():
        residual = load_residual_head(args.residual_checkpoint, device, latent_dim=latent_dim)

    planner = LatentPlanner(
        model,
        device,
        cost_fn="hover",
        residual_head=residual,
        planning=args.planner,
        cost_weights=MultiStepCostWeights(latent_smooth=args.latent_smooth),
    )
    context_frames = max(1, model.encoder.num_temporal // 2)

    raw_path = Path(args.capture_raw)
    video_path = Path(args.video)
    times = _load_raw_times(raw_path)
    stem = raw_path.name.replace(".raw.csv", "").replace(".csv", "")
    out_path = Path(args.out_dir) / f"{stem}_shadow.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    budget_ms = 1000.0 / float(args.agent_hz)
    buffer: deque[np.ndarray] = deque(maxlen=context_frames)
    fps_hint = float(args.fps_hint)

    with out_path.open("w") as f:
        for i, t_s in enumerate(times):
            loop0 = time.perf_counter()
            t_rgb0 = time.perf_counter()
            frame = _frame_at(video_path, float(t_s), fps_hint)
            if frame is None:
                rgb = np.zeros((3, img_size, img_size), dtype=np.float32)
            else:
                rgb = _resize_chw01(frame, img_size)
            t_rgb_ms = (time.perf_counter() - t_rgb0) * 1000.0
            buffer.append(rgb)
            control, cost, encode_ms = _propose(
                planner,
                buffer,
                context_frames=context_frames,
                residual_head=residual,
                hover_thrust=args.hover_thrust,
                num_candidates=args.num_candidates,
                action_scale=args.action_scale,
                seed=args.seed + i,
            )
            loop_ms = (time.perf_counter() - loop0) * 1000.0
            row = {
                "t": float(t_s),
                "vp": float(control[0]),
                "vq": float(control[1]),
                "vr": float(control[2]),
                "T": float(control[3]),
                "plan_cost": cost,
                "t_rgb_ms": t_rgb_ms,
                "t_encode_plan_ms": encode_ms,
                "loop_ms": loop_ms,
                "budget_ms": budget_ms,
                "source": "offline",
                "capture_raw": str(raw_path),
                "video": str(video_path),
            }
            f.write(json.dumps(row) + "\n")
            if (i + 1) % 25 == 0:
                print(f"[shadow] {i + 1}/{len(times)} t={t_s:.2f}s loop_ms={loop_ms:.1f}")
    return out_path


def run_live(args: argparse.Namespace) -> Path:
    from aerojepa.sim.tello_shadow import TelloShadowVehicle

    device = get_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    img_size = int(cfg["data"]["img_size"])
    latent_dim = int(cfg["encoder"]["embed_dim"])
    residual = None
    if args.residual_checkpoint and Path(args.residual_checkpoint).exists():
        residual = load_residual_head(args.residual_checkpoint, device, latent_dim=latent_dim)

    planner = LatentPlanner(
        model,
        device,
        cost_fn="hover",
        residual_head=residual,
        planning=args.planner,
        cost_weights=MultiStepCostWeights(latent_smooth=args.latent_smooth),
    )
    context_frames = max(1, model.encoder.num_temporal // 2)
    budget_ms = 1000.0 / float(args.agent_hz)
    period = 1.0 / float(args.agent_hz)

    stem = args.session or _session_stem(args.name_prefix, args.tags)
    out_path = Path(args.out_dir) / f"{stem}_shadow.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    vehicle = TelloShadowVehicle(
        img_size=img_size,
        min_battery=args.min_battery,
        high_res=args.high_res,
    )
    buffer: deque[np.ndarray] = deque(maxlen=context_frames)
    vehicle.reset()
    buffer.append(vehicle.rgb())

    print(f"[shadow] live observer -> {out_path} (duration {args.duration}s)")
    t_end = time.time() + float(args.duration)
    i = 0
    try:
        with out_path.open("w") as f:
            next_t = time.time()
            while time.time() < t_end:
                loop0 = time.perf_counter()
                t_rgb0 = time.perf_counter()
                obs = vehicle.poll()
                t_rgb_ms = (time.perf_counter() - t_rgb0) * 1000.0
                buffer.append(obs.rgb)
                control, cost, encode_ms = _propose(
                    planner,
                    buffer,
                    context_frames=context_frames,
                    residual_head=residual,
                    hover_thrust=args.hover_thrust,
                    num_candidates=args.num_candidates,
                    action_scale=args.action_scale,
                    seed=args.seed + i,
                )
                # Observer: never call vehicle.step(control).
                loop_ms = (time.perf_counter() - loop0) * 1000.0
                row = {
                    "t": float(obs.state.timestamp_s),
                    "unix_s": time.time(),
                    "vp": float(control[0]),
                    "vq": float(control[1]),
                    "vr": float(control[2]),
                    "T": float(control[3]),
                    "plan_cost": cost,
                    "t_rgb_ms": t_rgb_ms,
                    "t_encode_plan_ms": encode_ms,
                    "loop_ms": loop_ms,
                    "budget_ms": budget_ms,
                    "height_m": float(obs.state.xyz[2]),
                    "source": "live",
                }
                f.write(json.dumps(row) + "\n")
                f.flush()
                i += 1
                if i % 20 == 0:
                    print(
                        f"[shadow] n={i} t={row['t']:.2f}s "
                        f"loop_ms={loop_ms:.1f} budget_ms={budget_ms:.1f}"
                    )
                next_t += period
                sleep = next_t - time.time()
                if sleep > 0:
                    time.sleep(sleep)
    finally:
        vehicle.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/action_conditioned_wilds/latest.pt",
    )
    parser.add_argument(
        "--residual-checkpoint",
        default="checkpoints/action_residual_wilds/best.pt",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--planner", default="gradient", choices=["shooting", "gradient"])
    parser.add_argument("--latent-smooth", type=float, default=0.05)
    parser.add_argument("--num-candidates", type=int, default=12)
    parser.add_argument("--action-scale", type=float, default=0.04)
    parser.add_argument("--hover-thrust", type=float, default=DEFAULT_HOVER_THRUST)
    parser.add_argument("--agent-hz", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="data/flights")
    parser.add_argument("--min-battery", type=int, default=20)
    parser.add_argument("--high-res", action="store_true")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Reuse capture preflight checklist; exit.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Own the live stream (observer). Not simultaneous with capture on one client.",
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--name-prefix", default="shadow")
    parser.add_argument("--tags", nargs="*", default=[])
    parser.add_argument("--session", default=None, help="Fixed session stem for *_shadow.jsonl")
    parser.add_argument(
        "--capture-raw",
        default=None,
        help="Offline: path to capture *.raw.csv (align shadow rows on column t).",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Offline: capture mp4 paired with --capture-raw.",
    )
    parser.add_argument("--fps-hint", type=float, default=15.0)
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()

    print(SAFETY_BANNER)

    if args.preflight:
        from aerojepa.data.tello import preflight_check

        info = preflight_check(min_battery=args.min_battery)
        print("\n=== Tello preflight checklist (shadow) ===")
        print(f"  opencv     : {'OK' if info.get('opencv') else 'MISSING'}")
        print(f"  djitellopy : {'OK' if info.get('djitellopy') else 'MISSING'}")
        print(f"  Wi-Fi      : {'OK' if info.get('wifi_reachable') else 'FAIL'}")
        if info.get("battery") is not None:
            print(f"  battery    : {info['battery']}% (min {info['min_battery']}%)")
        print(f"\nOverall: {'PASS' if info.get('ok') else 'FAIL'}")
        sys.exit(0 if info.get("ok") else 1)

    if args.capture_raw and args.video:
        out = run_offline(args)
        print(f"\nDone. Shadow log: {out}")
        return

    if args.live:
        if not args.yes:
            reply = input("Pilot ready (observer only)? Type 'y': ").strip().lower()
            if reply != "y":
                print("Aborted.")
                sys.exit(1)
        out = run_live(args)
        print(f"\nDone. Shadow log: {out}")
        return

    print(
        "Specify offline (--capture-raw + --video) or --live. "
        "See --help. Prefer offline alignment to a capture raw CSV."
    )
    sys.exit(2)


if __name__ == "__main__":
    main()
