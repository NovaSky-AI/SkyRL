"""
Entrypoint for VLM RL training with VisGym.

Supports both recipes via ``--env_variant``:

    --env_variant sft       -> registers env_sft.VisGymEnv (tuple action format)
    --env_variant instruct  -> registers env_instruct.VisGymEnv (keyword action format)

Usage:
    uv run --isolated --extra fsdp \
        python examples/train/visgym/entrypoint.py \
        --env_variant instruct \
        generator.vision_language_generator=true [Hydra config overrides...]
"""

import argparse
import multiprocessing as mp
import sys

import ray

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl_gym.envs import register

mp.set_start_method("spawn", force=True)

_ENV_ENTRY_POINTS = {
    "sft": "examples.train.visgym.env_sft:VisGymEnv",
    "instruct": "examples.train.visgym.env_instruct:VisGymEnv",
}


@ray.remote(num_cpus=1)
def visgym_entrypoint(cfg: SkyRLTrainConfig, env_variant: str):
    register(id="visgym", entry_point=_ENV_ENTRY_POINTS[env_variant])

    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--env_variant", required=True, choices=list(_ENV_ENTRY_POINTS.keys()))
    args, remaining = parser.parse_known_args(sys.argv[1:])

    cfg = SkyRLTrainConfig.from_cli_overrides(remaining)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(visgym_entrypoint.remote(cfg, args.env_variant))


if __name__ == "__main__":
    main()
