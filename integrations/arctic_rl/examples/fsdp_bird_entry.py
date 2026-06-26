"""FSDP-native SkyRL entrypoint that registers the ``bird`` env on driver +
Ray workers. Stock ``main_base`` doesn't import the integration, so this shim
side-effect-imports ``integrations.arctic_rl.envs`` on the driver and arranges
for Ray workers to do the same. No core SkyRL changes.
"""

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[3]
sys.path.insert(0, str(_REPO_ROOT))
import integrations.arctic_rl.envs  # noqa: F401,E402  register `bird` on driver

import ray  # noqa: E402
import skyrl.train.entrypoints.main_base as _mb  # noqa: E402
import skyrl.train.utils.utils as _skyrl_utils  # noqa: E402

_repo_root_str = str(_REPO_ROOT)

_original_prepare = _skyrl_utils.prepare_runtime_environment


def _patched_prepare(cfg):
    env_vars = _original_prepare(cfg)
    existing_pp = env_vars.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    if _repo_root_str not in existing_pp.split(":"):
        env_vars["PYTHONPATH"] = (
            _repo_root_str + (":" + existing_pp if existing_pp else "")
        )
    if "SKYRL_USE_LIGER" in os.environ:
        env_vars["SKYRL_USE_LIGER"] = os.environ["SKYRL_USE_LIGER"]
    return env_vars


_skyrl_utils.prepare_runtime_environment = _patched_prepare


@ray.remote(num_cpus=1)
def _skyrl_entrypoint_with_bird(cfg):
    import sys as _sys

    if _repo_root_str not in _sys.path:
        _sys.path.insert(0, _repo_root_str)
    import integrations.arctic_rl.envs  # noqa: F401  register `bird` on worker

    exp = _mb.BasePPOExp(cfg)
    exp.run()


_mb.skyrl_entrypoint = _skyrl_entrypoint_with_bird


if __name__ == "__main__":
    _mb.main()
