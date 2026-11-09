"""The OPD trainer: ``RayPPOTrainer`` plus teacher scoring. Works with any ``GeneratorInterface``.

- ``generate_on_input_batch`` runs one ``generator.generate`` call per row concurrently and scores each row under
  the teacher as soon as its rollout is back, so only the last rows to finish wait on the teacher. With
  ``opd.topk_metrics`` on, the teacher also returns its own top-K on a few rows, which the overlap metrics compare
  with the student's captured top-K; the capture is dropped after. The base ``generate`` then logs the rollout
  metrics and validates the output, as for any trainer.
- ``postprocess_generator_output`` logs the truncation and repetition rates, records which rollouts are correct,
  and zeroes the rewards in pure mode, after their metrics are logged.
- ``convert_to_training_input`` pads the teacher logprobs into the batch, right-aligned like ``rollout_logprobs``,
  next to the correctness flags.
- ``compute_advantages_and_returns`` subtracts ``kl_coef * (action_log_probs - teacher_logprobs)`` on trainable
  tokens, after the configured advantage estimator and regulated by the ``opd`` stabilization settings, and logs
  the teacher's informativeness.
"""

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
        # One progress bar per row would flood the console. SkyRLGymGenerator.generate takes
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
        """Generate every row concurrently; score each under the teacher as soon as its rollout is back.

        Eval batches take the base path: one ``generator.generate`` call, no teacher.
        """
        batch_metadata = input_batch.get("batch_metadata")
        if batch_metadata is not None and batch_metadata.training_phase != "train":
            return await super().generate_on_input_batch(input_batch)

        rows = split_generator_input(input_batch)
        topk_cfg = self.cfg.trainer.algorithm.opd.topk_metrics
        # The rows the overlap metrics measure: num_rows evenly spaced ones, or every row when the batch is smaller.
        num_measured = min(topk_cfg.num_rows, len(rows))
        measured_rows = {(i * len(rows)) // num_measured for i in range(num_measured)}
        rollout_done_times: List[float] = []
        teacher_times: List[float] = []
        topk_row_stats: List[TopKStats] = []

        async def generate_and_score(row: GeneratorInput, measure_topk: bool) -> GeneratorOutput:
            # TODO (kyuds): current generator setup will return top-K support regardless of whether
            # this information is used to compute top-K metrics. We should develop a way to not return
            # unused data.
            output: GeneratorOutput = await self.generator.generate(row, **self._generate_kwargs)
            rollout_done = time.monotonic()
            score = await self.teacher_client.compute_logprobs(
                output["prompt_token_ids"][0],
                np.asarray(output["response_ids"][0], dtype=np.int32).reshape(-1, 1),
                topk=topk_cfg.top_k if measure_topk else 0,
            )
            output[TEACHER_LOGPROBS_KEY] = [score.logprobs[:, 0].tolist()]
            supports = output.pop(SAMPLE_SUPPORT_FIELD, None) if topk_cfg.num_rows else None
            if measure_topk:
                if supports is None or len(supports) != 1:
                    raise RuntimeError(
                        f"opd.topk_metrics needs one {SAMPLE_SUPPORT_FIELD!r} entry per trajectory from "
                        f"the generator, got {'none' if supports is None else len(supports)}"
                    )
                topk_row_stats.append(_row_topk_stats(supports[0], score, output["loss_masks"][0]))
            rollout_done_times.append(rollout_done)
            teacher_times.append(time.monotonic() - rollout_done)
            return output

        outputs = await tqdm.gather(
            *(generate_and_score(row, index in measured_rows) for index, row in enumerate(rows)),
            desc="Generating and scoring rows",
            miniters=max(1, len(rows) // 10),
            mininterval=5,
        )
        done = time.monotonic()
        self.all_metrics.update(sum(topk_row_stats, TopKStats()).metrics())
        self.all_metrics["opd/teacher_time_per_row_mean"] = (
            sum(teacher_times) / len(teacher_times) if teacher_times else 0.0
        )
        # Teacher latency is exposed when teachers are still scoring after generation has completed.
        self.all_metrics["opd/teacher_time_exposed"] = done - max(rollout_done_times) if rollout_done_times else 0.0
        # Concatenate output in input order.
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
