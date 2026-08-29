# Vehicle protocol (closed-loop)

Closed-loop episodes talk to a small `Vehicle` surface
(`reset` / `rgb` / `state` / `step` / `close`) in
`src/aerojepa/sim/vehicle.py`. Control is always `(vp, vq, vr, T)`
clipped to rates ±π and thrust `[0, 0.8]`.

`PyFlytVehicle` is the only adapter today (`PyFlyt/QuadX-Hover-v4`,
`agent_hz=40`). The heuristic `aerojepa_to_pyflyt` map stays on the
control side; gym / wind fields stay inside the adapter.

This is a research demo, not a flight controller. Tello capture
scripts remain record-only.
