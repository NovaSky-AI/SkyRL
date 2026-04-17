"""
Custom entrypoint for VLM RL training with VisGym environments.

Sets ``generator.is_vlm=True`` so that BasePPOExp wires in
SkyRLVLMGymGenerator (multi-modal) instead of the default
SkyRLGymGenerator (text-only).

Usage:
    uv run --isolated --extra fsdp \
        python examples/train/visgym/visgym_entrypoint.py \
        generator.is_vlm=True [config overrides...]
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
def visgym_entrypoint(cfg: SkyRLTrainConfig):
    register(
        id="visgym",  # <-- The name of the environment.
        entry_point="examples.train.visgym.env:VisGymEnv",  # <-- The path to the environment class.
    )

    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(visgym_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
