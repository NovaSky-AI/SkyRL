"""
Main entrypoint for generating rollouts on Harbor tasks. For debugging purposes.
"""

import sys

import ray
import asyncio
import yaml
from loguru import logger

from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.generators.base import GeneratorInput, TrajectoryID
from ..harbor_generator import HarborGenerator
from ..dataset import HarborTaskDataset
from .main_harbor import (
    HarborSkyRLConfig,
    HARBOR_DEFAULT_CONFIG,
    _deep_merge,
)


# For debugging purposes, we only generate a few samples.
NUM_SAMPLES_TO_TEST = 10


class HarborGenerateExp(BasePPOExp):
    def get_generator(self, cfg, tokenizer, inference_engine_client):
        """
        Initializes the HarborGenerator.
        """
        return HarborGenerator(
            generator_cfg=cfg.generator,
            harbor_cfg=cfg.harbor_trial_config,  # Pass harbor config to the generator
            inference_engine_client=inference_engine_client,
            tokenizer=tokenizer,
            max_seq_len=cfg.trainer.algorithm.max_seq_len,
        )

    def _setup_generator(self):
        logger.info(self.get_cfg_as_str(self.cfg))

        inference_engine_client = self.get_inference_client()
        asyncio.run(inference_engine_client.wake_up())

        return self.get_generator(self.cfg, self.tokenizer, inference_engine_client)

    def get_train_dataset(self):
        """Initializes the training dataset.

        Returns:
            HarborTaskDataset: The training dataset.
        """
        prompts_dataset = HarborTaskDataset(
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
def skyrl_entrypoint(cfg):
    # make sure that the training loop is not run on the head node.
    exp = HarborGenerateExp(cfg)
    exp.run()


def main() -> None:
    cfg = HarborSkyRLConfig.from_cli_overrides(sys.argv[1:])

    # Load harbor defaults and merge CLI overrides on top
    with open(HARBOR_DEFAULT_CONFIG) as f:
        defaults = yaml.safe_load(f)
    cfg.harbor_trial_config = _deep_merge(defaults, cfg.harbor_trial_config)

    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
