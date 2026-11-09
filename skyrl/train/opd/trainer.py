"""The OPD trainer: ``RayPPOTrainer`` plus teacher scoring. Works with any ``GeneratorInterface``.

- ``generate_on_input_batch`` runs one ``generator.generate`` call per prompt group concurrently and scores
  each group under the teacher as soon as its rollouts are back, so only the last groups to finish wait on the
  teacher. The base ``generate`` then logs the rollout metrics and validates the output, as for any trainer.
- ``postprocess_generator_output`` zeroes the rewards in pure mode, after their metrics are logged.
- ``convert_to_training_input`` pads the teacher logprobs into the batch, right-aligned like ``rollout_logprobs``.
- ``compute_advantages_and_returns`` subtracts ``kl_coef * (action_log_probs - teacher_logprobs)`` on trainable
  tokens, after the configured advantage estimator.
"""

import asyncio
import inspect
import time
from typing import List, Optional, Tuple

import torch
from tqdm.asyncio import tqdm

from skyrl.backends.skyrl_train.fields import (
    BatchMetadataKey,
    GeneratorInputKey,
    GeneratorOutputKey,
    TrainingField,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput
from skyrl.train.generators.utils import concatenate_generator_outputs
from skyrl.train.opd.teacher_client import TeacherLogprobClient
from skyrl.train.opd.utils import (
    TEACHER_LOGPROBS_KEY,
    apply_opd_to_advantages,
    pad_teacher_logprobs,
    split_generator_input,
)
from skyrl.train.trainer import RayPPOTrainer


class OPDTrainer(RayPPOTrainer):
    """``RayPPOTrainer`` for on-policy distillation. Works with any ``GeneratorInterface``."""

    def __init__(self, *args, teacher_client: Optional[TeacherLogprobClient], **kwargs):
        super().__init__(*args, **kwargs)
        assert teacher_client is not None
        self.teacher_client: TeacherLogprobClient = teacher_client
        # One progress bar per group would flood the console. SkyRLGymGenerator.generate takes
        # `disable_tqdm` for exactly this; other generators may not (same check as the fully-async trainer).
        self._generate_kwargs = (
            {"disable_tqdm": True} if "disable_tqdm" in inspect.signature(self.generator.generate).parameters else {}
        )

    async def train(self):
        """Close the teacher client's network sessions after the training loop terminates."""
        try:
            await super().train()
        finally:
            await self.teacher_client.aclose()

    async def generate_on_input_batch(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate every prompt's group concurrently; score each group under the teacher as it finishes.

        Eval batches take the base path: one ``generator.generate`` call, no teacher.
        """
        batch_metadata = input_batch.get(GeneratorInputKey.BATCH_METADATA)
        if batch_metadata is not None and batch_metadata.training_phase != "train":
            return await super().generate_on_input_batch(input_batch)

        groups = split_generator_input(input_batch, self.cfg.generator.n_samples_per_prompt)
        rollout_done_times: List[float] = []
        teacher_times: List[float] = []

        async def generate_and_score(group: GeneratorInput) -> GeneratorOutput:
            output: GeneratorOutput = await self.generator.generate(group, **self._generate_kwargs)
            rollout_done = time.monotonic()
            output[TEACHER_LOGPROBS_KEY] = list(
                await asyncio.gather(
                    *(
                        self.teacher_client.compute_logprobs(prompt_ids, response_ids)
                        for prompt_ids, response_ids in zip(
                            output[GeneratorOutputKey.PROMPT_TOKEN_IDS], output[GeneratorOutputKey.RESPONSE_IDS]
                        )
                    )
                )
            )
            rollout_done_times.append(rollout_done)
            teacher_times.append(time.monotonic() - rollout_done)
            return output

        outputs = await tqdm.gather(
            *(generate_and_score(group) for group in groups),
            desc="Generating and scoring groups",
            miniters=max(1, len(groups) // 10),
            mininterval=5,
        )
        done = time.monotonic()

        self.all_metrics["opd/teacher_time_per_group_mean"] = (
            sum(teacher_times) / len(teacher_times) if teacher_times else 0.0
        )
        # The only teacher cost the overlap cannot hide: the groups still being scored after the last
        # rollout of the batch came back.
        self.all_metrics["opd/teacher_time_exposed"] = done - max(rollout_done_times) if rollout_done_times else 0.0
        # One output in input order, with the rollout metrics re-aggregated over the whole batch and
        # `teacher_logprobs` kept as one more per-row list. `generate` logs and validates it.
        return concatenate_generator_outputs(outputs, step_wise=self.cfg.generator.step_wise_trajectories)

    @torch.no_grad()
    def postprocess_generator_output(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str]]:
        generator_output, uids = super().postprocess_generator_output(generator_output, uids)
        if not self.cfg.trainer.algorithm.opd.use_task_reward:
            # Pure OPD: the estimator sees zeros, so A_rl = 0 and the advantage is the teacher term
            # alone. super() already logged the verifier's pass rate (reward/avg_pass_at_n) from the
            # real rewards, which stays a free diagnostic for pure runs.
            generator_output[GeneratorOutputKey.REWARDS] = [
                [0.0] * len(reward) for reward in generator_output[GeneratorOutputKey.REWARDS]
            ]
        return generator_output, uids

    def convert_to_training_input(self, generator_output: GeneratorOutput, uids: List[str]) -> TrainingInputBatch:
        teacher_logprobs = generator_output.pop(TEACHER_LOGPROBS_KEY, None)
        if teacher_logprobs is None:
            raise RuntimeError(
                f"GeneratorOutput has no {TEACHER_LOGPROBS_KEY!r}: rollouts must come from "
                "OPDTrainer.generate_on_input_batch"
            )
        batch = super().convert_to_training_input(generator_output, uids)
        batch[TEACHER_LOGPROBS_KEY] = pad_teacher_logprobs(
            teacher_logprobs, batch[TrainingField.RESPONSE_MASK], batch.metadata.get(BatchMetadataKey.PAD_SIZE, 0)
        )
        return batch

    @torch.no_grad()
    def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
        data = super().compute_advantages_and_returns(data)  # the configured estimator; moves data to CPU
        teacher_logprobs: torch.Tensor = data.pop(TEACHER_LOGPROBS_KEY)  # never shipped to the workers
        action_log_probs = data[TrainingField.ACTION_LOG_PROBS]
        if action_log_probs is None:
            raise RuntimeError(
                "action_log_probs is None: the policy forward pass was skipped, but the OPD term needs the "
                "old-policy logprobs (see validate_opd_cfg)"
            )
        advantages, metrics = apply_opd_to_advantages(
            advantages=data[TrainingField.ADVANTAGES],
            action_log_probs=action_log_probs,
            teacher_logprobs=teacher_logprobs.to(action_log_probs.device),
            loss_mask=data[TrainingField.LOSS_MASK],
            kl_coef=self.cfg.trainer.algorithm.opd.kl_coef,
        )
        data[TrainingField.ADVANTAGES] = advantages
        self.all_metrics.update(metrics)
        return data
