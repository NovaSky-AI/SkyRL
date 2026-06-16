"""
Main entrypoint for the LLM-as-a-judge example.
"""

import sys

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils import initialize_ray
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    # Register the llm_as_a_judge environment inside the entrypoint task (no need to modify the skyrl-gym package).
    register(
        id="llm_as_a_judge",
        entry_point="examples.train.llm_as_a_judge.llm_judge_env:GSM8kLLMJudgeEnv",
    )

    # make sure that the training loop is not run on the head node.
    exp = BasePPOExp(cfg)
    exp.run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
