"""SkyRL entrypoint for the CholecT50 GRPO (RLVR) run.

Mirrors ``examples/train/geometry3k/geometry3k_entrypoint.py``: register the
custom env class, then run the standard SkyRL PPO/GRPO experiment from the
Hydra-style CLI overrides in ``run_cholec_neuron.sh``.
"""
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
def cholec_entrypoint(cfg: SkyRLTrainConfig):
    register(
        id="cholec",
        entry_point="examples.train.cholec.env:CholecEnv",
    )
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(cholec_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
