from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from aerojepa.data.telemetry import (
    ACTION_COLUMNS,
    RAW_STATE_COLUMNS,
    derive_actions_from_raw,
)

# DJI Tello capture.
#
# The Tello is the project's primary real-world test platform. This module
# records the camera stream and the drone's reported state, then writes them in
# the exact layout ``VideoClipDataset`` expects, with timestamped filenames.
#
# SAFETY BY DESIGN: this code NEVER commands takeoff, movement, or landing. It
# only turns the video stream on, reads frames + telemetry, and turns the stream
# off. A human pilot flies the aircraft manually (Tello app or controller) while
# this records. Keeping capture and control separate means a software bug here
# can never move the drone.
#
# ``djitellopy`` is an optional, hardware-only dependency:
#     pip install djitellopy opencv-python


@dataclass
class CaptureConfig:
    """Options for a single Tello capture session."""

    out_dir: str | Path = "data/flights"
    duration_s: float = 30.0
    fps: int = 15
    name_prefix: str = "flight"
    min_battery: int = 20          # abort below this %, so we never fly on fumes
    high_res: bool = False         # Tello 720p (True) vs lower-latency stream
    record_telemetry: bool = True
    tags: list[str] = field(default_factory=list)  # optional filename annotations


@dataclass
class CaptureResult:
    video_path: Path
    action_csv_path: Path | None
    raw_csv_path: Path | None
    num_frames: int
    duration_s: float
    fps: float


def _timestamped_name(prefix: str, tags: list[str]) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [prefix, stamp, *tags]
    return "_".join(p for p in parts if p)


def _read_state_row(drone, t0: float) -> list[float]:
    """Snapshot the drone's current telemetry into a RAW_STATE_COLUMNS row.

    Missing/failed readings default to 0.0 so one flaky sensor never aborts a
    capture. Order must match ``RAW_STATE_COLUMNS``.
    """
    def safe(fn) -> float:
        try:
            return float(fn())
        except Exception:  # noqa: BLE001 - hardware readings are best-effort
            return 0.0

    return [
        time.time() - t0,
        safe(drone.get_speed_x),
        safe(drone.get_speed_y),
        safe(drone.get_speed_z),
        safe(drone.get_height),
        safe(drone.get_yaw),
        safe(drone.get_pitch),
        safe(drone.get_roll),
        safe(drone.get_battery),
        safe(drone.get_distance_tof),
        safe(drone.get_barometer),
    ]


def _write_csv(path: Path, header: tuple[str, ...], rows: np.ndarray) -> None:
    with path.open("w") as f:
        f.write(",".join(header) + "\n")
        for row in np.atleast_2d(rows):
            f.write(",".join(f"{v:.6f}" for v in row) + "\n")


def capture_flight(config: CaptureConfig | None = None, **overrides) -> CaptureResult:
    """Record one Tello flight clip (+ telemetry) to ``out_dir``.

    Reads the live camera and telemetry for ``duration_s`` seconds at ``fps``,
    then writes a timestamped ``.mp4`` plus (optionally) two CSVs: a training-
    ready ``<name>.csv`` (ACTION_COLUMNS) and a ``<name>.raw.csv`` provenance log.

    Never commands flight -- fly manually while this records. Raises clear errors
    if the optional deps or a drone are missing, or if the battery is too low.
    """
    config = config or CaptureConfig()
    for k, v in overrides.items():
        setattr(config, k, v)

    try:
        from djitellopy import Tello  # type: ignore
    except ImportError as exc:  # pragma: no cover - hardware-only path
        raise ImportError(
            "Tello capture needs djitellopy. On the ground-station machine run "
            "`pip install djitellopy opencv-python`."
        ) from exc
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - hardware-only path
        raise ImportError("Tello capture needs opencv-python (`pip install opencv-python`).") from exc

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = _timestamped_name(config.name_prefix, config.tags)

    drone = Tello()
    drone.connect()
    battery = drone.get_battery()
    print(f"[tello] connected. battery={battery}%")
    if battery < config.min_battery:
        raise RuntimeError(
            f"Battery {battery}% is below --min-battery {config.min_battery}%. "
            "Charge before capturing; low battery risks an uncontrolled landing."
        )

    if config.high_res:
        try:
            drone.set_video_resolution(Tello.RESOLUTION_720P)
        except Exception:  # noqa: BLE001 - not all firmwares support this
            print("[tello] warning: could not set 720p; using default stream.")

    frames: list[np.ndarray] = []
    raw_rows: list[list[float]] = []
    t0 = time.time()
    period = 1.0 / max(1, config.fps)

    drone.streamon()
    try:
        reader = drone.get_frame_read()
        print(f"[tello] recording {config.duration_s:.0f}s @ {config.fps}fps -> {name}.mp4")
        next_t = time.time()
        while time.time() - t0 < config.duration_s:
            frame = reader.frame
            if frame is None:
                time.sleep(0.005)
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if config.record_telemetry:
                raw_rows.append(_read_state_row(drone, t0))
            next_t += period
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
    finally:
        # Always stop the stream, even on Ctrl-C or a mid-flight error.
        try:
            drone.streamoff()
        except Exception:  # noqa: BLE001
            pass

    if not frames:
        raise RuntimeError("No frames captured -- is the video stream up and the drone powered?")

    elapsed = time.time() - t0
    real_fps = len(frames) / max(1e-6, elapsed)
    height, width = frames[0].shape[:2]

    video_path = out_dir / f"{name}.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), config.fps, (width, height)
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    action_csv = raw_csv = None
    if config.record_telemetry and raw_rows:
        raw = np.asarray(raw_rows, dtype=np.float32)
        raw_csv = out_dir / f"{name}.raw.csv"
        _write_csv(raw_csv, RAW_STATE_COLUMNS, raw)
        action_csv = out_dir / f"{name}.csv"
        _write_csv(action_csv, ACTION_COLUMNS, derive_actions_from_raw(raw))

    print(
        f"[tello] saved {len(frames)} frames ({elapsed:.1f}s, {real_fps:.1f} real fps) "
        f"-> {video_path}"
    )
    return CaptureResult(
        video_path=video_path,
        action_csv_path=action_csv,
        raw_csv_path=raw_csv,
        num_frames=len(frames),
        duration_s=elapsed,
        fps=real_fps,
    )


def ping_tello_host(host: str = "192.168.10.1", timeout_s: float = 1.5) -> bool:
    """Return True if the Tello Wi-Fi AP responds to a single ping."""
    import platform
    import subprocess

    if platform.system().lower() == "windows":
        cmd = ["ping", "-n", "1", "-w", str(int(timeout_s * 1000)), host]
    else:
        wait = max(1, int(timeout_s))
        cmd = ["ping", "-c", "1", "-W", str(wait), host]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, timeout=timeout_s + 2.0, check=False,
        )
        return proc.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False


def preflight_check(min_battery: int = 20, tello_host: str = "192.168.10.1") -> dict:
    """Run a full ground-station checklist before capture. No recording, no flight.

    Checks optional deps, Wi-Fi reachability to the Tello AP, then connects and
    reads battery/temperature/state before disconnecting cleanly.
    """
    checklist: dict[str, bool | int | float | str | None] = {
        "tello_host": tello_host,
        "min_battery": min_battery,
    }

    try:
        import cv2  # noqa: F401
        checklist["opencv"] = True
    except ImportError:
        checklist["opencv"] = False

    try:
        from djitellopy import Tello  # type: ignore
        checklist["djitellopy"] = True
    except ImportError:
        checklist["djitellopy"] = False

    checklist["wifi_reachable"] = ping_tello_host(tello_host)
    checklist["battery"] = None
    checklist["temperature_c"] = None
    checklist["height_cm"] = None
    checklist["barometer"] = None
    checklist["battery_ok"] = False

    deps_ok = bool(checklist["opencv"] and checklist["djitellopy"])
    if not deps_ok or not checklist["wifi_reachable"]:
        checklist["ok"] = False
        return checklist

    drone = Tello()
    try:
        drone.connect()
        battery = int(drone.get_battery())
        checklist["battery"] = battery
        checklist["temperature_c"] = float(drone.get_temperature())
        checklist["height_cm"] = float(drone.get_height())
        checklist["barometer"] = float(drone.get_barometer())
        checklist["battery_ok"] = battery >= min_battery
    finally:
        try:
            drone.end()
        except Exception:  # noqa: BLE001
            pass

    checklist["ok"] = bool(checklist["battery_ok"])
    return checklist
