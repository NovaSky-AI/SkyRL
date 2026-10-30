"""Neuron (Trainium2) multi-core smoke entrypoint for the ``multiply`` text env.

Mirrors ``examples/train/cholec/cholec_entrypoint.py`` exactly -- the only
neuron-specific change from ``main_multiply.py`` is ``mp.set_start_method("spawn")``
(the neuron runtime cannot be forked).  Registers the multiply env in-process and runs
the standard ``BasePPOExp`` GRPO experiment from Hydra-style CLI overrides in
``run_multiply_neuron.sh``.
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
def multiply_neuron_entrypoint(cfg: SkyRLTrainConfig):
    register(
        id="multiply",
        entry_point="examples.train.multiply.env:MultiplyEnv",
    )
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(multiply_neuron_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
