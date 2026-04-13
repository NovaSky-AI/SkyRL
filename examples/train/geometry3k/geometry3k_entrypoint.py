import multiprocessing as mp
import sys

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl_gym.envs import register

mp.set_start_method("spawn", force=True)


@ray.remote(num_cpus=1)
def geometry3k_entrypoint(cfg: SkyRLTrainConfig):
    register(
        id="geometry3k",
        entry_point="examples.train.geometry3k.env:Geometry3kEnv",
    )

    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(geometry3k_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
