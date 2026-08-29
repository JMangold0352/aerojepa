"""Tello shadow observer (Vehicle-shaped; never commands the aircraft).

Implements ``reset`` / ``rgb`` / ``state`` / ``close``. ``step`` always raises.
Stream + telemetry read only - same safety posture as ``aerojepa.data.tello``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from aerojepa.sim.vehicle import VehicleObs, VehicleState, clip_control


class TelloShadowVehicle:
    """Observer-only Tello adapter for shadow planning logs.

    Does not implement a control path. Calling :meth:`step` is a hard error.
    """

    def __init__(
        self,
        *,
        img_size: int = 64,
        min_battery: int = 20,
        high_res: bool = False,
    ) -> None:
        self.img_size = int(img_size)
        self.min_battery = int(min_battery)
        self.high_res = bool(high_res)
        self._drone: Any = None
        self._reader: Any = None
        self._t0 = 0.0
        self._last: VehicleObs | None = None

    def reset(self, seed: int | None = None) -> VehicleObs:
        del seed
        try:
            from djitellopy import Tello  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Tello shadow needs djitellopy (`pip install djitellopy opencv-python`)."
            ) from exc
        try:
            import cv2  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Tello shadow needs opencv-python (`pip install opencv-python`)."
            ) from exc

        self.close()
        drone = Tello()
        drone.connect()
        battery = int(drone.get_battery())
        if battery < self.min_battery:
            try:
                drone.end()
            except Exception:  # noqa: BLE001
                pass
            raise RuntimeError(
                f"Battery {battery}% is below min-battery {self.min_battery}%."
            )
        if self.high_res:
            try:
                drone.set_video_resolution(Tello.RESOLUTION_720P)
            except Exception:  # noqa: BLE001
                pass
        drone.streamon()
        self._drone = drone
        self._reader = drone.get_frame_read()
        self._t0 = time.time()
        for _ in range(10):
            frame = self._reader.frame
            if frame is not None:
                break
            time.sleep(0.05)
        self._last = self._pack()
        return self._last

    def rgb(self) -> np.ndarray:
        if self._last is None:
            raise RuntimeError("call reset() before rgb()")
        return self._last.rgb

    def state(self) -> VehicleState:
        if self._last is None:
            raise RuntimeError("call reset() before state()")
        return self._last.state

    def step(self, control: np.ndarray) -> VehicleObs:
        del control
        raise RuntimeError(
            "TelloShadowVehicle is observer-only: step() is refused. "
            "Log proposed controls; never apply them."
        )

    def close(self) -> None:
        self._reader = None
        drone, self._drone = self._drone, None
        self._last = None
        if drone is None:
            return
        try:
            drone.streamoff()
        except Exception:  # noqa: BLE001
            pass
        try:
            drone.end()
        except Exception:  # noqa: BLE001
            pass

    def poll(self) -> VehicleObs:
        """Refresh rgb + state from the live stream (no control)."""
        self._last = self._pack()
        return self._last

    def _pack(self) -> VehicleObs:
        if self._drone is None or self._reader is None:
            raise RuntimeError("TelloShadowVehicle is not connected; call reset()")
        import cv2

        frame = self._reader.frame
        if frame is None:
            rgb = np.zeros((3, self.img_size, self.img_size), dtype=np.float32)
        else:
            rgb_u8 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = _resize_chw01(rgb_u8, self.img_size)

        st = _vehicle_state_from_drone(self._drone, t0=self._t0)
        return VehicleObs(rgb=rgb, state=st)


def _resize_chw01(rgb_u8: np.ndarray, img_size: int) -> np.ndarray:
    import cv2

    h, w = rgb_u8.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = rgb_u8[y0 : y0 + side, x0 : x0 + side]
    if crop.shape[0] != img_size:
        crop = cv2.resize(crop, (img_size, img_size), interpolation=cv2.INTER_AREA)
    out = crop.astype(np.float32) / 255.0
    return np.transpose(out, (2, 0, 1))


def _safe_float(fn) -> float:
    try:
        return float(fn())
    except Exception:  # noqa: BLE001
        return 0.0


def _vehicle_state_from_drone(drone: Any, *, t0: float) -> VehicleState:
    """Build a VehicleState from Tello readouts (observer; no world XY).

    Tello does not expose a world-frame XY here. ``xyz`` is therefore
    ``(0, 0, height_m)`` with ``height_m = get_height() / 100`` (cm → m).
    ``vxyz`` is ``get_speed_{x,y,z}() / 100`` (cm/s → m/s); those speeds are
    not integrated into a dead-reckoned trajectory. Do not treat logged
    ``xyz`` as a metric path for waypoint success.
    """
    vgx = _safe_float(drone.get_speed_x) / 100.0
    vgy = _safe_float(drone.get_speed_y) / 100.0
    vgz = _safe_float(drone.get_speed_z) / 100.0
    height_m = _safe_float(drone.get_height) / 100.0
    yaw = np.deg2rad(_safe_float(drone.get_yaw))
    pitch = np.deg2rad(_safe_float(drone.get_pitch))
    roll = np.deg2rad(_safe_float(drone.get_roll))
    return VehicleState(
        xyz=np.array([0.0, 0.0, height_m], dtype=np.float32),
        vxyz=np.array([vgx, vgy, vgz], dtype=np.float32),
        euler_rad=np.array([roll, pitch, yaw], dtype=np.float32),
        rates_rad=None,
        timestamp_s=float(time.time() - t0),
    )


__all__ = ["TelloShadowVehicle", "clip_control"]
