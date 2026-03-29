# Fleet Task Environment Integration for SkyRL
#
# This module provides a SkyRL-compatible environment wrapper for Fleet-hosted tasks.
# It uses OpenEnv's FleetTaskEnv as the abstraction layer.

__all__ = ["FleetTaskEnv"]


def __getattr__(name: str):
    """Lazy import to avoid import errors when dependencies are not installed."""
    if name == "FleetTaskEnv":
        from skyrl_gym.envs.fleet_task.env import FleetTaskEnv

        return FleetTaskEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
