import sys

from skyrl_train.config import SkyRLConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_train.utils import initialize_ray
import ray
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLConfig):
    # Register the openenv environment inside the entrypoint task (no need to modify the skyrl-gym package).
    register(
        id="openenv",
        entry_point="integrations.openenv.env:OpenEnv",
    )
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLConfig.from_cli_overrides(sys.argv[1:])
    # Validate config args.
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
