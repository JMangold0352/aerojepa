from __future__ import annotations

from typing import Any, Callable

# Optional simulator hooks. Thin adapters to common quadrotor simulators.
# Not required for core training/eval. Install extras (`PyFlyt`,
# `gym-pybullet-drones`) and import lazily so the base install stays lean.


def make_pyflyt_env(env_id: str = "PyFlyt/QuadX-Hover-v4", **kwargs: Any):
    """Create a PyFlyt Gymnasium environment, if PyFlyt is installed.

    PyFlyt is a lightweight, well-maintained UAV simulator that is easy to run
    headless -- our preferred target for latent-space planning experiments.
    Defaults to ``QuadX-Hover-v4`` (matches the AeroProber data generator).
    """
    try:
        import gymnasium as gym
        import PyFlyt.gym_envs  # noqa: F401  (registers the environments)
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "PyFlyt simulation needs `pip install PyFlyt gymnasium`. "
            "This is an optional extra; the core project does not require it."
        ) from exc
    return gym.make(env_id, **kwargs)


def make_pybullet_drones_env(**kwargs: Any):
    """Create a gym-pybullet-drones environment, if it is installed."""
    try:
        import gym_pybullet_drones  # noqa: F401
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "gym-pybullet-drones simulation needs `pip install gym-pybullet-drones gymnasium`. "
            "This is an optional extra; the core project does not require it."
        ) from exc
    return gym.make("hover-aviary-v0", **kwargs)


def run_hover_episode(
    policy: Callable[[Any], Any] | None = None,
    env_id: str = "PyFlyt/QuadX-Hover-v4",
    max_steps: int = 500,
    seed: int = 0,
    **env_kwargs: Any,
) -> dict[str, Any]:
    """Run one closed-loop hover episode in PyFlyt with a pluggable ``policy``.

    This is the integration seam between AeroJEPA's planner and a real quadrotor
    simulator: pass a ``policy(observation) -> action`` callable (for example one
    that wraps :class:`aerojepa.sim.planner.LatentPlanner`) and this drives the
    environment, returning the trajectory and total reward. With no policy it
    applies zero action -- a smoke test that the simulator is wired up.

    Kept dependency-light: PyFlyt/gymnasium are imported lazily inside
    :func:`make_pyflyt_env`, so importing this module never requires them.

    Returns a dict with ``rewards`` (per step), ``total_reward``, ``steps``, and
    ``observations`` (the recorded trajectory).
    """
    env = make_pyflyt_env(env_id, **env_kwargs)
    obs, _info = env.reset(seed=seed)

    observations = [obs]
    rewards: list[float] = []
    total = 0.0
    for _ in range(max_steps):
        if policy is None:
            action = env.action_space.sample() * 0.0  # inert baseline
        else:
            action = policy(obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        observations.append(obs)
        rewards.append(float(reward))
        total += float(reward)
        if terminated or truncated:
            break
    env.close()

    return {
        "rewards": rewards,
        "total_reward": total,
        "steps": len(rewards),
        "observations": observations,
    }
