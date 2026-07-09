"""AeroProber: physics-inspired prober on top of the frozen AeroJEPA world model.

The prober (psi) converts latent rollouts from the frozen encoder + (looped)
predictor into physically meaningful metric states (position, velocity, Euler
attitude, angular velocity) through a differentiable kinematic integrator.

Design choices for v1 (see research/prober/README.md):
- Attitude: Euler angles (yaw, pitch, roll), wrapped to (-180, 180].
- Ground-truth metric state: PyFlyt quadrotor simulator (real dynamics).
- Predictor staging: regular (single-pass) first; looped arm added later.
- Encoder + predictor are frozen; only the prober is trained.
"""

from aerojepa_research.prober.integrator import (
    GRAVITY_Z,
    ControlIntegrator,
    KinematicIntegrator,
    MetricState,
    wrap_degrees,
)
from aerojepa_research.prober.prober import CONTROL_DIM, PlainMLPHead, Prober, pool_latents

__all__ = [
    "CONTROL_DIM",
    "GRAVITY_Z",
    "ControlIntegrator",
    "KinematicIntegrator",
    "MetricState",
    "PlainMLPHead",
    "Prober",
    "pool_latents",
    "wrap_degrees",
]