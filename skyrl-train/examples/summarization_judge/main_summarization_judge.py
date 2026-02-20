"""
Main entrypoint for the summarization LLM-as-a-judge example.
"""

import ray
import hydra
from omegaconf import DictConfig
from skyrl_train.utils import initialize_ray
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_gym.envs import register


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # Register the summarization_judge environment inside the entrypoint task
    register(
        id="summarization_judge",
        entry_point="examples.summarization_judge.summarization_judge_env:SummarizationJudgeEnv",
    )

    # Run the training
    exp = BasePPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Validate the config
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
