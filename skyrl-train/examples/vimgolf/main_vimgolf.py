"""
uv run --isolated --extra vllm -m examples.vimgolf.main_vimgolf
"""

import hydra
import ray
from omegaconf import DictConfig
from skyrl_gym.envs import register

from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # Register the VimGolf environments inside the entrypoint task (no need to modify the skyrl-gym package).
    register(
        id="vimgolf-single-turn",
        entry_point="examples.vimgolf.env:VimGolfSingleTurnEnv",
    )

    register(
        id="vimgolf-multi-turn",
        entry_point="examples.vimgolf.env:VimGolfMultiTurnEnv",
    )

    register(
        id="vimgolf-gym",
        entry_point="examples.vimgolf.env:VimGolfGymEnv",
    )

    # make sure that the training loop is not run on the head node.
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
