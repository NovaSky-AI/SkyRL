"""Registers Arctic-RL-shipped envs with skyrl-gym at integration-import time.

Uses ``__name__`` for the entry_point so registration works under any import
path (e.g. ``integrations.arctic_rl.envs`` from core dispatch, or ``arctic_rl.envs``
when the integration dir is on PYTHONPATH).

Both ``bird`` and ``bird_sql`` resolve to the same env: launcher recipes pass
``environment.env_class=bird`` for evals, while preprocessed BIRD parquets
have ``env_class="bird_sql"`` baked per-row (this is the verl PR #6 schema).
Registering both names keeps either source-of-truth working without a parquet
rewrite.
"""

from skyrl_gym.envs.registration import register

for _id in ("bird", "bird_sql"):
    register(id=_id, entry_point=f"{__name__}.bird:BirdEnv")

__all__ = []
