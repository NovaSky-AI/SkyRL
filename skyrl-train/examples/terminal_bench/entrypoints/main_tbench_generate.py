"""
Main entrypoint for generating rollouts on terminal bench tasks. For debugging purposes.
"""

import ray
import asyncio
import hydra
from loguru import logger
from omegaconf import DictConfig

from skyrl_train.utils import validate_cfg
from skyrl_train.utils.utils import initialize_ray
from skyrl_train.entrypoints.main_base import (
    BasePPOExp,
    config_dir,
)
from skyrl_train.generators.base import GeneratorInput, TrajectoryID
from examples.terminal_bench.generator.terminal_bench_generator import TerminalBenchGenerator
from examples.terminal_bench.dataset import TerminalBenchTaskDataset


# For debugging purposes, we only generate a few samples.
NUM_SAMPLES_TO_TEST = 10


class TerminalBenchGenerateExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the TerminalBenchGenerator.
        """
        return TerminalBenchGenerator(
            generator_cfg=cfg.generator,
            terminal_bench_cfg=cfg.terminal_bench_config,  # Pass terminal_bench config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
        )

    def _setup_generator(self):
        logger.info(self.get_cfg_as_str(self.cfg))

        inference_engine_client = self.get_inference_client()
        asyncio.run(inference_engine_client.wake_up())

        return self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            TerminalBenchTaskDataset: The training dataset.
        """
        prompts_dataset = TerminalBenchTaskDataset(
            data_files=self.cfg.data.train_data,
        )
        assert (
            len(prompts_dataset) >= self.cfg.trainer.train_batch_size
        ), f"dataset should be atleast as large as `train_batch_size` {self.cfg.trainer.train_batch_size}, got size {len(prompts_dataset)}"
        return prompts_dataset

    def run(self):
        generator = self._setup_generator()

        prompts = []
        trajectory_ids = []
        for item in self.train_dataset:
            prompts.append(item["prompt"])
            trajectory_ids.append(TrajectoryID(instance_id=item["uid"], repetition_id=0))

        # Build input from the training dataset
        input_batch = GeneratorInput(
            prompts=prompts[:NUM_SAMPLES_TO_TEST],
            trajectory_ids=trajectory_ids[:NUM_SAMPLES_TO_TEST],
            env_classes=None,
            env_extras=None,
            sampling_params=None,
        )

        # Start generation
        asyncio.run(generator.generate(input_batch))


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    # make sure that the training loop is not run on the head node.
    exp = TerminalBenchGenerateExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    # validate the arguments
    validate_cfg(cfg)

    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
