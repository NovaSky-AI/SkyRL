"""The OPD trainer: ``RayPPOTrainer`` plus teacher scoring. Works with any ``GeneratorInterface``.

- ``generate_on_input_batch`` runs one ``generator.generate`` call per prompt group concurrently and scores
  each group under the teacher as soon as its rollouts are back, so only the last groups to finish wait on the
  teacher. With ``opd.top_k > 0`` the teacher also scores the student's captured candidates, which then ride on
  ``rollout_sample_support``. The base ``generate`` then logs the rollout metrics and validates the output, as for
  any trainer.
- ``postprocess_generator_output`` logs the truncation and repetition rates, records which rollouts are correct,
  and zeroes the rewards in pure mode, after their metrics are logged.
- ``convert_to_training_input`` pads the teacher logprobs into the batch, right-aligned like ``rollout_logprobs``,
  next to the correctness flags.
- ``compute_advantages_and_returns`` subtracts ``kl_coef * (action_log_probs - teacher_logprobs)`` on trainable
  tokens, after the configured advantage estimator and regulated by the ``opd`` stabilization settings, and logs
  the teacher's informativeness. With ``opd.top_k > 0`` it adds TT-OPD's top-K term: each position's ``C`` on the
  sampled token's advantage, and the candidates' logit weights in the support's member-value column.
- ``_normalize_advantages`` gives those logit weights the per-token loss-reduction scale the advantages get.
"""

import asyncio
import inspect
import time
from typing import List, Optional, Tuple

import torch
from tqdm.asyncio import tqdm

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    SampleSupportColumn,
)
from skyrl.train.generators.base import GeneratorInput, GeneratorOutput
from skyrl.train.generators.utils import concatenate_generator_outputs
from skyrl.train.opd.metrics import (
    informativeness,
    is_repetitive,
    rollout_correct,
    truncation_rate,
)
from skyrl.train.opd.teacher_client import TeacherLogprobClient
from skyrl.train.opd.utils import (
    ROLLOUT_CORRECT_KEY,
    STUDENT_LOGPROB_FLOOR,
    TEACHER_LOGPROBS_KEY,
    apply_opd_to_advantages,
    candidate_matrix,
    member_rows,
    pad_teacher_logprobs,
    row_cells,
    split_generator_input,
    topk_metrics,
    topk_weights,
    with_member_values,
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

        groups = split_generator_input(input_batch, self.cfg.generator.n_samples_per_prompt)
        top_k = self.cfg.trainer.algorithm.opd.top_k
        rollout_done_times: List[float] = []
        teacher_times: List[float] = []

        async def generate_and_score(group: GeneratorInput) -> GeneratorOutput:
            output: GeneratorOutput = await self.generator.generate(group, **self._generate_kwargs)
            rollout_done = time.monotonic()
            # Only top_k > 0 captures candidates; any other support (e.g. replay's) is left alone.
            supports = output.get(SAMPLE_SUPPORT_FIELD) if top_k else None
            if top_k and supports is None:
                raise RuntimeError(
                    f"opd.top_k > 0 needs {SAMPLE_SUPPORT_FIELD!r} from the generator, which returned none"
                )
            responses = output["response_ids"]
            candidates = [
                candidate_matrix(response_ids, support)
                for response_ids, support in zip(
                    responses, supports if supports is not None else [None] * len(responses), strict=True
                )
            ]
            scores = await asyncio.gather(
                *(
                    self.teacher_client.compute_logprobs(prompt_ids, row)
                    for prompt_ids, row in zip(output["prompt_token_ids"], candidates, strict=True)
                )
            )
            output[TEACHER_LOGPROBS_KEY] = [score.logprobs[:, 0].tolist() for score in scores]
            if supports is not None:
                # Rows [sampled, candidates...] with the teacher's logprobs as the member values.
                output[SAMPLE_SUPPORT_FIELD] = [
                    support.with_teacher_scores(row, score.logprobs)
                    for support, row, score in zip(supports, candidates, scores, strict=True)
                ]
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
        opd = self.cfg.trainer.algorithm.opd
        topk = topk_weights(data[SAMPLE_SUPPORT_FIELD], data["loss_mask"], opd) if opd.top_k else None
        advantages, metrics = apply_opd_to_advantages(
            advantages=data["advantages"],
            action_log_probs=action_log_probs,
            teacher_logprobs=teacher_logprobs,
            loss_mask=data["loss_mask"],
            opd=opd,
            sampled_in_topk=None if topk is None else topk.sampled_in_topk,
            topk_weight_sum=None if topk is None else topk.token_weights,
        )
        value = informativeness(
            teacher_logprobs - action_log_probs.clamp(min=STUDENT_LOGPROB_FLOOR),
            data["loss_mask"],
            correct,
        )
        if value is not None:
            metrics["opd/informativeness"] = value
        data["advantages"] = advantages
        if topk is not None:
            # From here on the member values are logit weights, not teacher logprobs.
            data[SAMPLE_SUPPORT_FIELD] = with_member_values(data[SAMPLE_SUPPORT_FIELD], topk.member_weights)
            metrics.update(topk_metrics(topk, data["loss_mask"]))
        self.all_metrics.update(metrics)
        return data

    @torch.no_grad()
    def _normalize_advantages(
        self,
        data: TrainingInputBatch,
        mini_batch_boundaries: List[Tuple[int, int]],
        prompt_boundaries: Optional[List[Tuple[int, int]]] = None,
    ) -> TrainingInputBatch:
        """With ``opd.top_k > 0``, also scale each logit weight by its token's loss-reduction scale.

        The worker sums the logit term like the policy loss, so the weights need the factor the reduction applies to
        each token's advantage: the reduction of a tensor of ones. That factor is the advantages' whole normalization,
        since every reduction is linear and OPD refuses ``advantage_batch_normalize``.
        """
        data = super()._normalize_advantages(data, mini_batch_boundaries, prompt_boundaries)
        if not self.cfg.trainer.algorithm.opd.top_k:
            return data
        support, loss_mask = data[SAMPLE_SUPPORT_FIELD], data["loss_mask"]
        scale = self._apply_loss_reduction(
            torch.ones_like(data["advantages"]), loss_mask, mini_batch_boundaries, prompt_boundaries
        )
        batch, column = row_cells(support, loss_mask.shape[1])
        weights = support.columns[SampleSupportColumn.MEMBER_VALUES] * scale[batch, column][member_rows(support)]
        data[SAMPLE_SUPPORT_FIELD] = with_member_values(support, weights)
        return data
