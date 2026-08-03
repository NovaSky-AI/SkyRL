"""Conversion from exact TITO traces to SkyRL ``GeneratorOutput``."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, TypedDict

from skyrl.train.generators.base import GeneratorOutput, TrajectoryID
from skyrl.train.generators.utils import get_rollout_metrics

from .types import CommittedTurn, TraceOutcome

logger = logging.getLogger(__name__)


class _TrainingRow(TypedDict):
    prompt_token_ids: List[int]
    response_ids: List[int]
    loss_mask: List[int]
    rollout_logprobs: List[float]
    stop_reason: str
    turn_start: int
    turn_end: int


def _is_prefix(prefix: Sequence[int], candidate: Sequence[int]) -> bool:
    return len(prefix) <= len(candidate) and list(prefix) == list(candidate[: len(prefix)])


def _merge_exact_turns(turns: Sequence[CommittedTurn]) -> List[_TrainingRow]:
    if not turns:
        return []

    rows: List[_TrainingRow] = []
    prompt_ids = list(turns[0].prompt_token_ids)
    response_ids = list(turns[0].completion_ids)
    loss_mask = [1] * len(response_ids)
    rollout_logprobs = list(turns[0].completion_logprobs)
    stop_reason = turns[0].stop_reason
    turn_start = 0
    turn_end = 1

    def flush() -> None:
        rows.append(
            {
                "prompt_token_ids": prompt_ids.copy(),
                "response_ids": response_ids.copy(),
                "loss_mask": loss_mask.copy(),
                "rollout_logprobs": rollout_logprobs.copy(),
                "stop_reason": stop_reason,
                "turn_start": turn_start,
                "turn_end": turn_end,
            }
        )

    for turn_index, turn in enumerate(turns[1:], start=1):
        merged_prefix = prompt_ids + response_ids
        if not _is_prefix(merged_prefix, turn.prompt_token_ids):
            flush()
            prompt_ids = list(turn.prompt_token_ids)
            response_ids = list(turn.completion_ids)
            loss_mask = [1] * len(response_ids)
            rollout_logprobs = list(turn.completion_logprobs)
            stop_reason = turn.stop_reason
            turn_start = turn_index
            turn_end = turn_index + 1
            continue

        prompt_delta = list(turn.prompt_token_ids[len(merged_prefix) :])
        response_ids.extend(prompt_delta)
        response_ids.extend(turn.completion_ids)
        loss_mask.extend([0] * len(prompt_delta))
        loss_mask.extend([1] * len(turn.completion_ids))
        rollout_logprobs.extend([0.0] * len(prompt_delta))
        rollout_logprobs.extend(turn.completion_logprobs)
        stop_reason = turn.stop_reason
        turn_end = turn_index + 1

    flush()
    return rows


def build_trace_generator_output(
    outcomes: Sequence[TraceOutcome],
    *,
    overlong_filtering: bool,
    step_wise: bool,
) -> GeneratorOutput:
    """Build training rows from exact committed turns."""
    merged_rows = [_merge_exact_turns(outcome.trace.committed_turns()) for outcome in outcomes]
    masked_instance_ids = {
        outcome.trajectory_id.instance_id
        for outcome in outcomes
        if outcome.stop_reason in ("agent_timeout", "error") or not outcome.trace.committed_turns()
    }
    if not step_wise:
        for outcome, rows in zip(outcomes, merged_rows):
            if len(rows) != 1:
                logger.warning(
                    "Masking trajectory %s because it produced %d exact token sequences "
                    "with step_wise_trajectories disabled",
                    outcome.trajectory_id.to_string(),
                    len(rows),
                )
                masked_instance_ids.add(outcome.trajectory_id.instance_id)

    prompt_token_ids: List[List[int]] = []
    response_ids: List[List[int]] = []
    rewards: List[float] = []
    loss_masks: List[List[int]] = []
    stop_reasons: List[str] = []
    rollout_logprobs: List[List[float]] = []
    trajectory_ids: List[TrajectoryID] = []
    is_last_step: List[bool] = []
    trajectory_generation_times: List[Optional[float]] = []

    successful_sequences: List[List[int]] = []
    successful_rewards: List[float] = []
    successful_times: List[Optional[float]] = []
    successful_turn_counts: List[int] = []
    has_routed_experts = False
    rollout_expert_indices: List[List[List[List[int]]]] = []

    for outcome, rows in zip(outcomes, merged_rows):
        trajectory_id = outcome.trajectory_id
        if trajectory_id.instance_id in masked_instance_ids:
            prompt_token_ids.append([0])
            response_ids.append([0])
            rewards.append(0.0)
            loss_masks.append([0])
            stop_reasons.append("error")
            rollout_logprobs.append([0.0])
            trajectory_ids.append(trajectory_id)
            is_last_step.append(True)
            trajectory_generation_times.append(outcome.generation_time)
            rollout_expert_indices.append([])
            continue

        for row_index, row in enumerate(rows):
            last = row_index == len(rows) - 1
            row_response_ids = list(row["response_ids"])
            row_loss_mask = list(row["loss_mask"])
            if outcome.stop_reason == "context_length" and overlong_filtering:
                row_loss_mask = [0] * len(row_loss_mask)

            prompt_token_ids.append(list(row["prompt_token_ids"]))
            response_ids.append(row_response_ids)
            rewards.append(outcome.reward if last else 0.0)
            loss_masks.append(row_loss_mask)
            stop_reasons.append(outcome.stop_reason)
            rollout_logprobs.append(list(row["rollout_logprobs"]))
            trajectory_ids.append(trajectory_id)
            is_last_step.append(last)
            trajectory_generation_times.append(outcome.generation_time)
            final_turn = outcome.trace.committed_turns()[row["turn_end"] - 1]
            row_experts: List[List[List[int]]] = []
            if final_turn.routed_experts is not None:
                expected_length = len(row["prompt_token_ids"]) + len(row["response_ids"])
                if len(final_turn.routed_experts) != expected_length:
                    raise ValueError("Routed-expert data does not align with the exact training sequence")
                has_routed_experts = True
                row_experts = [[list(layer) for layer in token] for token in final_turn.routed_experts]
            rollout_expert_indices.append(row_experts)

        successful_sequences.append(list(rows[-1]["prompt_token_ids"]) + list(rows[-1]["response_ids"]))
        successful_rewards.append(outcome.reward)
        successful_times.append(outcome.generation_time)
        successful_turn_counts.append(outcome.num_turns)

    metric_times = (
        None
        if any(value is None for value in successful_times)
        else [float(value) for value in successful_times if value is not None]
    )
    output_times = (
        None
        if any(value is None for value in trajectory_generation_times)
        else [float(value) for value in trajectory_generation_times if value is not None]
    )
    rollout_metrics = (
        get_rollout_metrics(
            successful_sequences,
            successful_rewards,
            trajectory_completion_times=metric_times,
        )
        if successful_sequences
        else {}
    )
    rollout_metrics["generate/trajectories_context_length_exceeded"] = sum(
        1 for outcome in outcomes if outcome.stop_reason == "context_length"
    )
    rollout_metrics["generate/avg_num_turns"] = (
        sum(successful_turn_counts) / len(successful_turn_counts) if successful_turn_counts else 0.0
    )
    rollout_metrics["generate/num_timeout_trajectories"] = sum(
        1 for outcome in outcomes if outcome.stop_reason == "agent_timeout"
    )
    rollout_metrics["generate/num_error_trajectories"] = sum(
        1 for outcome in outcomes if outcome.stop_reason == "error" or not outcome.trace.committed_turns()
    )
    rollout_metrics["generate/num_masked_instances"] = len(masked_instance_ids)

    return GeneratorOutput(
        prompt_token_ids=prompt_token_ids,
        response_ids=response_ids,
        rewards=rewards,
        loss_masks=loss_masks,
        stop_reasons=stop_reasons,
        rollout_metrics=rollout_metrics,
        rollout_logprobs=rollout_logprobs,
        trajectory_ids=trajectory_ids,
        trajectory_generation_times=output_times,
        trajectory_time_splits=None,
        rollout_expert_indices=rollout_expert_indices if has_routed_experts else None,
        is_last_step=is_last_step if step_wise else None,
        env_metrics=None,
        pixel_values=None,
        image_grid_thw=None,
    )
