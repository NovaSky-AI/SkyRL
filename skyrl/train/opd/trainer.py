"""``OPDTrainer``: ``RayPPOTrainer`` plus teacher scoring and the consumers of ``teacher_logprobs``.

Nothing here touches the generator. ``generate`` splits the batch into one ``GeneratorInput`` per
prompt (its ``n_samples_per_prompt`` rows) and runs one ``generator.generate`` call per group
concurrently, the way ``FullyAsyncRayPPOTrainer._run_generate_for_a_group_loop`` does; each group
is scored under the teacher as soon as its rollouts are back, while the other groups are still
generating. Only the groups that finish last pay for the teacher. This works with any
``GeneratorInterface``, which matters because users bring their own generator for custom harnesses.

The other three overrides consume the resulting ``GeneratorOutput["teacher_logprobs"]``:

- ``postprocess_generator_output`` zeroes the per-token rewards in pure mode, after their metrics
  are logged, so the advantage estimator contributes nothing.
- ``convert_to_training_input`` pads the teacher logprobs into the batch, right-aligned like
  ``rollout_logprobs``.
- ``compute_advantages_and_returns`` subtracts ``kl_coef * (action_log_probs - teacher_logprobs)``
  on trainable tokens *after* the configured estimator, which is what lets the teacher term compose
  with any estimator in mixed mode and any reward-only estimator in pure mode (GRPO would otherwise
  sum the per-token signal into one scalar per sequence).
"""

import asyncio
import inspect
import time
from typing import List, Tuple

import torch
from tqdm.asyncio import tqdm

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
from skyrl.train.utils.trainer_utils import validate_generator_output


class OPDTrainer(RayPPOTrainer):
    """``RayPPOTrainer`` for on-policy distillation. Works with any ``GeneratorInterface``."""

    def __init__(self, *args, teacher_client: TeacherLogprobClient, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_client = teacher_client
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

    @torch.no_grad()
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        """Generate every prompt's group concurrently; score each group under the teacher as it finishes.

        Eval batches take the base path: one ``generator.generate`` call, no teacher.
        """
        batch_metadata = input_batch.get("batch_metadata")
        if batch_metadata is not None and batch_metadata.training_phase != "train":
            return await super().generate(input_batch)

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
                        for prompt_ids, response_ids in zip(output["prompt_token_ids"], output["response_ids"])
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

        # Same tail as RayPPOTrainer.generate. concatenate_generator_outputs re-aggregates the rollout
        # metrics over the whole batch (the fully-async trainer's path) and keeps `teacher_logprobs`
        # as one more per-row list.
        step_wise = self.cfg.generator.step_wise_trajectories
        generator_output = concatenate_generator_outputs(outputs, step_wise=step_wise)
        if generator_output["rollout_metrics"] is not None:
            self.all_metrics.update(generator_output["rollout_metrics"])
        generator_output.pop("rollout_metrics", None)
        validate_generator_output(len(input_batch["prompts"]), generator_output, step_wise=step_wise)

        self.all_metrics["opd/teacher_time_per_group_mean"] = (
            sum(teacher_times) / len(teacher_times) if teacher_times else 0.0
        )
        # The only teacher cost the overlap cannot hide: the groups still being scored after the last
        # rollout of the batch came back.
        self.all_metrics["opd/teacher_time_exposed"] = done - max(rollout_done_times) if rollout_done_times else 0.0
        return generator_output

    @torch.no_grad()
    def postprocess_generator_output(
        self, generator_output: GeneratorOutput, uids: List[str]
    ) -> Tuple[GeneratorOutput, List[str]]:
        generator_output, uids = super().postprocess_generator_output(generator_output, uids)
        if not self.cfg.trainer.algorithm.opd.use_task_reward:
            # Pure OPD: the estimator sees zeros, so A_rl = 0 and the advantage is the teacher term
            # alone. super() already logged the verifier's pass rate (reward/avg_pass_at_n) from the
            # real rewards, which stays a free diagnostic for pure runs.
            generator_output["rewards"] = [[0.0] * len(reward) for reward in generator_output["rewards"]]
        return generator_output, uids

    def convert_to_training_input(self, generator_output: GeneratorOutput, uids: List[str]) -> TrainingInputBatch:
        teacher_logprobs = generator_output.pop(TEACHER_LOGPROBS_KEY, None)
        if teacher_logprobs is None:
            raise RuntimeError(
                f"GeneratorOutput has no {TEACHER_LOGPROBS_KEY!r}: rollouts must come from OPDTrainer.generate"
            )
        batch = super().convert_to_training_input(generator_output, uids)
        batch[TEACHER_LOGPROBS_KEY] = pad_teacher_logprobs(
            teacher_logprobs, batch["response_mask"], batch.metadata.get("pad_size", 0)
        )
        return batch

    @torch.no_grad()
    def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
        data = super().compute_advantages_and_returns(data)  # the configured estimator; moves data to CPU
        teacher_logprobs: torch.Tensor = data.pop(TEACHER_LOGPROBS_KEY)  # never shipped to the workers
        action_log_probs = data["action_log_probs"]
        if action_log_probs is None:
            raise RuntimeError(
                "action_log_probs is None: the policy forward pass was skipped, but the OPD term needs the "
                "old-policy logprobs (see validate_opd_cfg)"
            )
        advantages, metrics = apply_opd_to_advantages(
            advantages=data["advantages"],
            action_log_probs=action_log_probs,
            teacher_logprobs=teacher_logprobs.to(action_log_probs.device),
            loss_mask=data["loss_mask"],
            kl_coef=self.cfg.trainer.algorithm.opd.kl_coef,
        )
        data["advantages"] = advantages
        self.all_metrics.update(metrics)
        return data
