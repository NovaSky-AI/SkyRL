"""The OPD trainer: ``RayPPOTrainer`` plus teacher scoring. Works with any ``GeneratorInterface``.

- ``generate_on_input_batch`` runs one ``generator.generate`` call per prompt group concurrently and scores
  each group under the teacher as soon as its rollouts are back, so only the last groups to finish wait on the
  teacher. With ``opd.topk_metrics`` on, the teacher also returns its own top-K on a few rows, which the overlap
  metrics compare with the student's captured top-K; the captured rows are dropped after. The base ``generate``
  then logs the rollout metrics and validates the output, as for any trainer.
- ``postprocess_generator_output`` logs the truncation and repetition rates, records which rollouts are correct,
  and zeroes the rewards in pure mode, after their metrics are logged.
- ``convert_to_training_input`` pads the teacher logprobs into the batch, right-aligned like ``rollout_logprobs``,
  next to the correctness flags.
- ``compute_advantages_and_returns`` subtracts ``kl_coef * (action_log_probs - teacher_logprobs)`` on trainable
  tokens, after the configured advantage estimator and regulated by the ``opd`` stabilization settings, and logs
  the teacher's informativeness.
"""

import asyncio
import inspect
import time
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm.asyncio import tqdm

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    CompactSampleSupport,
)
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput
from skyrl.train.generators.utils import concatenate_generator_outputs
from skyrl.train.opd.metrics import (
    TopKStats,
    informativeness,
    is_repetitive,
    rollout_correct,
    topk_stats,
    truncation_rate,
)
from skyrl.train.opd.teacher_client import CandidateScores, TeacherLogprobClient
from skyrl.train.opd.utils import (
    ROLLOUT_CORRECT_KEY,
    STUDENT_LOGPROB_FLOOR,
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
        batch_metadata = input_batch.get("batch_metadata")
        if batch_metadata is not None and batch_metadata.training_phase != "train":
            return await super().generate_on_input_batch(input_batch)

        group_size = self.cfg.generator.n_samples_per_prompt
        groups = split_generator_input(input_batch, group_size)
        topk_cfg = self.cfg.trainer.algorithm.opd.topk_metrics
        # Rows the overlap metrics measure, by batch index, evenly spaced; every row when the batch is smaller.
        # Group g holds rows g * group_size onward.
        num_total_rows = len(input_batch["prompts"])
        num_measured = min(topk_cfg.num_rows, num_total_rows)
        topk_rows = {(i * num_total_rows) // num_measured for i in range(num_measured)}
        rollout_done_times: List[float] = []
        teacher_times: List[float] = []
        topk_group_stats: List[TopKStats] = []

        async def generate_and_score(group_index: int, group: GeneratorInput) -> GeneratorOutput:
            output: GeneratorOutput = await self.generator.generate(group, **self._generate_kwargs)
            rollout_done = time.monotonic()
            # The overlap metrics' capture is read here and never trained on, so it leaves the output.
            supports = output.pop(SAMPLE_SUPPORT_FIELD, None) if topk_cfg.num_rows else None
            if topk_cfg.num_rows and supports is None:
                raise RuntimeError(
                    f"opd.topk_metrics needs {SAMPLE_SUPPORT_FIELD!r} from the generator, which returned none"
                )
            responses = output["response_ids"]
            if supports is not None and len(supports) != len(responses):
                raise ValueError(f"{len(supports)} {SAMPLE_SUPPORT_FIELD!r} entries for {len(responses)} trajectories")
            rows = [group_index * group_size + i for i in range(len(responses))]
            scores = await asyncio.gather(
                *(
                    self.teacher_client.compute_logprobs(
                        prompt_ids,
                        np.asarray(response_ids, dtype=np.int32).reshape(-1, 1),
                        topk=topk_cfg.top_k if row in topk_rows else 0,
                    )
                    for row, prompt_ids, response_ids in zip(rows, output["prompt_token_ids"], responses, strict=True)
                )
            )
            output[TEACHER_LOGPROBS_KEY] = [score.logprobs[:, 0].tolist() for score in scores]
            if supports is not None:
                topk_group_stats.append(
                    sum(
                        (
                            _row_topk_stats(support, score, loss_mask)
                            for row, support, score, loss_mask in zip(
                                rows, supports, scores, output["loss_masks"], strict=True
                            )
                            if row in topk_rows
                        ),
                        TopKStats(),
                    )
                )
            rollout_done_times.append(rollout_done)
            teacher_times.append(time.monotonic() - rollout_done)
            return output

        outputs = await tqdm.gather(
            *(generate_and_score(group_index, group) for group_index, group in enumerate(groups)),
            desc="Generating and scoring groups",
            miniters=max(1, len(groups) // 10),
            mininterval=5,
        )
        done = time.monotonic()
        if topk_group_stats:
            self.all_metrics.update(sum(topk_group_stats, TopKStats()).metrics())

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
        generator_output[ROLLOUT_CORRECT_KEY] = [rollout_correct(reward) for reward in generator_output["rewards"]]
        # Metrics: Repetition Rate and Truncation Rate
        responses = generator_output["response_ids"]
        self.all_metrics["opd/repetition_rate"] = sum(map(is_repetitive, responses)) / len(responses)
        if generator_output.get("stop_reasons") is not None:
            self.all_metrics["opd/truncation_rate"] = truncation_rate(generator_output["stop_reasons"])
        # Pure OPD: advantage is from the teacher alone. Rewards are set to 0.
        if not self.cfg.trainer.algorithm.opd.use_task_reward:
            generator_output["rewards"] = [[0.0] * len(reward) for reward in generator_output["rewards"]]
        return generator_output, uids

    def convert_to_training_input(self, generator_output: GeneratorOutput, uids: List[str]) -> TrainingInputBatch:
        teacher_logprobs = generator_output.pop(TEACHER_LOGPROBS_KEY, None)
        if teacher_logprobs is None:
            raise RuntimeError(
                f"GeneratorOutput has no {TEACHER_LOGPROBS_KEY!r}: rollouts must come from "
                "OPDTrainer.generate_on_input_batch"
            )
        correct = generator_output.pop(ROLLOUT_CORRECT_KEY, None)
        if correct is None:
            raise RuntimeError(
                f"GeneratorOutput has no {ROLLOUT_CORRECT_KEY!r}: rollouts must go through "
                "OPDTrainer.postprocess_generator_output"
            )
        batch = super().convert_to_training_input(generator_output, uids)
        pad_size = batch.metadata.get("pad_size", 0)
        batch[TEACHER_LOGPROBS_KEY] = pad_teacher_logprobs(teacher_logprobs, batch["response_mask"], pad_size)
        # Padding rows have an all-zero loss mask, so informativeness never counts their flag.
        batch[ROLLOUT_CORRECT_KEY] = torch.tensor(correct + [False] * pad_size)
        return batch

    @torch.no_grad()
    def compute_advantages_and_returns(self, data: TrainingInputBatch) -> TrainingInputBatch:
        data = super().compute_advantages_and_returns(data)  # the configured estimator; moves data to CPU
        teacher_logprobs: torch.Tensor = data.pop(TEACHER_LOGPROBS_KEY)  # never shipped to the workers
        correct: torch.Tensor = data.pop(ROLLOUT_CORRECT_KEY)  # never shipped to the workers
        action_log_probs = data["action_log_probs"]
        if action_log_probs is None:
            raise RuntimeError(
                "action_log_probs is None: the policy forward pass was skipped, but the OPD term needs the "
                "old-policy logprobs (see validate_opd_cfg)"
            )
        teacher_logprobs = teacher_logprobs.to(action_log_probs.device)
        advantages, metrics = apply_opd_to_advantages(
            advantages=data["advantages"],
            action_log_probs=action_log_probs,
            teacher_logprobs=teacher_logprobs,
            loss_mask=data["loss_mask"],
            opd=self.cfg.trainer.algorithm.opd,
        )
        value = informativeness(
            teacher_logprobs - action_log_probs.clamp(min=STUDENT_LOGPROB_FLOOR),
            data["loss_mask"],
            correct,
        )
        if value is not None:
            metrics["opd/informativeness"] = value
        data["advantages"] = advantages
        self.all_metrics.update(metrics)
        return data


def _row_topk_stats(support: CompactSampleSupport, scores: CandidateScores, loss_mask: List[int]) -> TopKStats:
    """One row's overlap sums. A token with no captured members (an appended EOS, an observation token) has no
    student top-K and is left out."""
    if len(support) != len(loss_mask):
        raise ValueError(f"{len(support)} support rows for {len(loss_mask)} response tokens")
    mask = np.asarray(loss_mask, dtype=bool) & (support.row_lengths > 0)
    return topk_stats(support.candidate_grid(), support.logprob_grid(), scores.topk_ids, scores.topk_logprobs, mask)
