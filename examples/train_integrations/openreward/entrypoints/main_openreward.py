"""SkyRL entrypoint for OpenReward environment training."""

import os
import sys

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.utils import initialize_ray
import ray
from skyrl_gym.envs import register

# Capture environment variables before Ray workers are spawned
_OPENREWARD_API_KEY = os.environ.get("OPENREWARD_API_KEY", "")
_OPENREWARD_UPLOAD_ROLLOUT = os.environ.get("OPENREWARD_UPLOAD_ROLLOUT", "true")
_OPENREWARD_RUN_NAME = os.environ.get("OPENREWARD_RUN_NAME", "skyrl-openreward")


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig, api_key: str, upload_rollout: str, run_name: str):
    # Propagate environment variables into this worker and any child processes
    os.environ["OPENREWARD_API_KEY"] = api_key
    os.environ["OPENREWARD_UPLOAD_ROLLOUT"] = upload_rollout
    os.environ["OPENREWARD_RUN_NAME"] = run_name

    register(
        id="openreward",
        entry_point="examples.train_integrations.openreward.env:OpenRewardEnv",
    )

    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg, _OPENREWARD_API_KEY, _OPENREWARD_UPLOAD_ROLLOUT, _OPENREWARD_RUN_NAME))


if __name__ == "__main__":
    main()
