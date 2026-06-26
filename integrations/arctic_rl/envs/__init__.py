"""Registers Arctic-RL-shipped envs with skyrl-gym at integration-import time.

Uses ``__name__`` for the entry_point so registration works under any import
path (e.g. ``integrations.arctic_rl.envs`` from core dispatch, or ``arctic_rl.envs``
when the integration dir is on PYTHONPATH).
"""

from skyrl_gym.envs.registration import register

register(
    id="bird",
    entry_point=f"{__name__}.bird:BirdEnv",
)

__all__ = []
