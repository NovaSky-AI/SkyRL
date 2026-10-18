import random
from typing import List, Tuple

from loguru import logger

from skyrl.train.generators.base import GeneratorOutput
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.utils import Timer


def make_dummy_generator_output(
    rng: random.Random,
    vocab_size: int,
    num_prompts: int,
    n_samples_per_prompt: int,
    prompt_length: int,
    response_length: int,
) -> Tuple[GeneratorOutput, List[str]]:
    """Builds a batch of max-length sequences with a random token id at every position.

    Token ids are drawn uniformly from ``[0, vocab_size)``, which spreads MoE routing across
    experts. Each prompt is shared by the ``n_samples_per_prompt`` responses of its group, and
    every response is drawn independently.

    Args:
        rng: Random number generator used for the token ids and rewards.
        vocab_size: Number of token ids to draw from.
        num_prompts: Number of prompts (groups) in the batch.
        n_samples_per_prompt: Number of responses per prompt.
        prompt_length: Number of tokens in each prompt.
        response_length: Number of tokens in each response.

    Returns:
        The generator output with ``num_prompts * n_samples_per_prompt`` samples, and the uid of
        each sample's group.
    """
    num_samples = num_prompts * n_samples_per_prompt
    prompt_token_ids = []
    uids = []
    for i in range(num_prompts):
        prompt = rng.choices(range(vocab_size), k=prompt_length)
        prompt_token_ids.extend([prompt] * n_samples_per_prompt)
        uids.extend([str(i)] * n_samples_per_prompt)
    response_ids = [rng.choices(range(vocab_size), k=response_length) for _ in range(num_samples)]
    generator_output = {
        "prompt_token_ids": prompt_token_ids,
        "response_ids": response_ids,
        "rewards": [[0] * (response_length - 1) + [rng.randint(0, 1)]] * num_samples,
        "loss_masks": [[1] * response_length] * num_samples,
    }
    return generator_output, uids


class FullCtxTrainer(RayPPOTrainer):
    """A dummy trainer that tests configurations with max sequence length.

    This trainer is meant to help users validate their configuration setup by:
    1. Creating max length sequences directly
    2. Running a few training steps

    This helps catch OOM issues early before running full training.
    """

    async def train(self):
        """Run a few training steps with max sequence length."""
        logger.info("Starting dummy training with max sequence length...")

        self.global_step = 0

        # Initialize weight sync state
        with Timer("init_weight_sync_state", self.all_timings):
            self.init_weight_sync_state()

        # Run a few training steps
        self.global_step += 1  # start from 1
        # Seeded so that runs with the same config train on the same dummy batches.
        rng = random.Random(self.cfg.trainer.seed)
        self._profiler_start()
        try:
            for step in range(self.cfg.trainer.num_dummy_steps):
                logger.info(f"Running dummy training step {step + 1}/{self.cfg.trainer.num_dummy_steps}")

                # Create the dummy batch with max-length sequences. It is built outside the step timer.
                dummy_generator_output, uids = make_dummy_generator_output(
                    rng=rng,
                    vocab_size=self.tokenizer.vocab_size,
                    num_prompts=self.cfg.trainer.train_batch_size,
                    n_samples_per_prompt=self.cfg.generator.n_samples_per_prompt,
                    prompt_length=self.cfg.generator.max_input_length,
                    response_length=self.cfg.generator.sampling_params.max_generate_length,
                )

                # Run a single training step
                with Timer("step", self.all_timings):
                    training_input = self.convert_to_training_input(dummy_generator_output, uids)

                    with Timer("fwd_logprobs_values_reward", self.all_timings):
                        training_input = self.fwd_logprobs_values_reward(training_input)

                    # 1.5 apply kl divergence penalty to rewards
                    if self.cfg.trainer.algorithm.use_kl_in_reward:
                        with Timer("apply_reward_kl_penalty", self.all_timings):
                            training_input = self.apply_reward_kl_penalty(training_input)

                    # 3. calculate advantages and returns
                    with Timer("compute_advantages_and_returns", self.all_timings):
                        training_input = self.compute_advantages_and_returns(training_input)
                        # remove some unwanted keys
                        for key in ["rewards"]:
                            training_input.pop(key)
                        training_input.metadata.pop("uids")

                    # 4. train policy/critic model
                    with Timer("train_critic_and_policy", self.all_timings):
                        status = self.train_critic_and_policy(training_input)

                    # One profiler step per global step.
                    self._profiler_step()

                    self.tracker.log(self.all_metrics, step=self.global_step)
                    self.all_metrics = {}
                    self.tracker.log(
                        {"timing/" + k: v for k, v in self.all_timings.items()},
                        step=self.global_step,
                    )
                    self.all_timings = {}
                    self.global_step += 1

                    logger.info(f"Step {step + 1} completed. Status: {status}")
        finally:
            self._profiler_stop()

        self.tracker.finish()
        logger.info("Dummy training completed successfully!")
