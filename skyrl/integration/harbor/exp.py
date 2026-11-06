"""The experiments Harbor's entrypoints run: training, fully-async training, and a few rollouts for debugging."""

import asyncio

from loguru import logger

from skyrl.integration.harbor.dataset import HarborTaskDataset
from skyrl.integration.harbor.generator import HarborGenerator
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.generators.base import GeneratorInput, TrajectoryID

# For debugging purposes, we only generate a few samples.
NUM_SAMPLES_TO_TEST = 10


class HarborExp(BasePPOExp):
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

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            HarborTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = HarborTaskDataset(
                data_files=self.cfg.data.val_data,
            )
            return prompts_dataset
        return None


class HarborFullyAsyncExp(HarborExp):
    """``HarborExp`` with ``FullyAsyncRayPPOTrainer``: the moral equivalent of ``examples/train/fully_async`` for Harbor."""

    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ):
        return FullyAsyncRayPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )


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

    def get_eval_dataset(self):
        """Initializes the evaluation dataset.

        Returns:
            HarborTaskDataset: The evaluation dataset.
        """
        if self.cfg.trainer.eval_interval > 0 and self.cfg.data.val_data:
            prompts_dataset = HarborTaskDataset(
                data_files=self.cfg.data.val_data,
            )
            return prompts_dataset
        return None

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
