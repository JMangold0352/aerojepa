# Vehicle protocol (closed-loop)

Closed-loop episodes talk to a small `Vehicle` surface
(`reset` / `rgb` / `state` / `step` / `close`) in
`src/aerojepa/sim/vehicle.py`. Control is always `(vp, vq, vr, T)`
clipped to rates ±π and thrust `[0, 0.8]`.

`PyFlytVehicle` is the only adapter today (`PyFlyt/QuadX-Hover-v4`,
`agent_hz=40`). The heuristic `aerojepa_to_pyflyt` map stays on the
control side; gym / wind fields stay inside the adapter.

## Timing and watchdog

Each step logs `t_rgb_ms`, `t_encode_plan_ms`, `t_step_ms`, `loop_ms`, and
`budget_ms` (= 1000 / `agent_hz`, 25 ms at 40 Hz). If RGB is bad, control is
NaN/Inf, or the loop so far exceeds 2× budget, the episode steps
last-good / hover hold instead of garbage and increments `watchdog_holds`.
This is a research hold on a laptop sim, not a dual-CPU flight watchdog.

This is a research demo, not a flight controller. Tello capture
scripts remain record-only.
