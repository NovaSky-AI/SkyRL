"""Conversion from exact TITO graph branches to SkyRL ``GeneratorOutput``."""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, TypedDict

from skyrl.train.generators.base import GeneratorOutput, TrajectoryID
from skyrl.train.generators.utils import get_rollout_metrics

from .trace import BranchView
from .types import TraceOutcome

logger = logging.getLogger(__name__)


class _TrainingRow(TypedDict):
    prompt_token_ids: List[int]
    response_ids: List[int]
    loss_mask: List[int]
    rollout_logprobs: List[float]
    rollout_expert_indices: Optional[List[List[List[int]]]]


def _branch_training_rows(branches: Sequence[BranchView]) -> List[_TrainingRow]:
    """Materialize graph leaves while training every sampled node at most once."""
    rows: List[_TrainingRow] = []
    seen_sampled_nodes: set[int] = set()

    for branch in branches:
        token_ids: List[int] = []
        trainable_mask: List[bool] = []
        logprobs: List[float] = []

        for node_id, node in zip(branch.node_ids, branch.nodes):
            token_ids.extend(node.token_ids)
            if node.sampled_start is not None and node_id in seen_sampled_nodes:
                trainable_mask.extend([False] * len(node.token_ids))
            else:
                trainable_mask.extend(node.sampled_mask)
                if any(node.sampled_mask):
                    seen_sampled_nodes.add(node_id)
            logprobs.extend(node.logprobs)

        try:
            first_trainable = trainable_mask.index(True)
        except ValueError:
            continue

        routed = branch.routed_experts
        rows.append(
            {
                "prompt_token_ids": token_ids[:first_trainable],
                "response_ids": token_ids[first_trainable:],
                "loss_mask": [int(value) for value in trainable_mask[first_trainable:]],
                "rollout_logprobs": logprobs[first_trainable:],
                "rollout_expert_indices": (
                    [[list(layer) for layer in token] for token in routed] if routed is not None else None
                ),
            }
        )
    return rows


def build_trace_generator_output(
    outcomes: Sequence[TraceOutcome],
    *,
    overlong_filtering: bool,
    step_wise: bool,
) -> GeneratorOutput:
    """Build one training row per trainable root-to-leaf graph branch."""
    branch_rows = [_branch_training_rows(outcome.trace.branches()) for outcome in outcomes]
    masked_instance_ids = {
        outcome.trajectory_id.instance_id
        for outcome, rows in zip(outcomes, branch_rows)
        if outcome.stop_reason in ("agent_timeout", "error") or not rows
    }
    if not step_wise:
        for outcome, rows in zip(outcomes, branch_rows):
            if len(rows) != 1:
                logger.warning(
                    "Masking trajectory %s because it produced %d trainable branches "
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
    rollout_expert_indices: List[List[List[List[int]]]] = []
    has_routed_experts = False

    successful_sequences: List[List[int]] = []
    successful_rewards: List[float] = []
    successful_times: List[Optional[float]] = []
    successful_turn_counts: List[int] = []
    successful_branch_counts: List[int] = []

    for outcome, rows in zip(outcomes, branch_rows):
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
            row_loss_mask = list(row["loss_mask"])
            if outcome.stop_reason == "context_length" and overlong_filtering:
                row_loss_mask = [0] * len(row_loss_mask)

            prompt_token_ids.append(list(row["prompt_token_ids"]))
            response_ids.append(list(row["response_ids"]))
            rewards.append(outcome.reward if last else 0.0)
            loss_masks.append(row_loss_mask)
            stop_reasons.append(outcome.stop_reason)
            rollout_logprobs.append(list(row["rollout_logprobs"]))
            trajectory_ids.append(trajectory_id)
            is_last_step.append(last)
            trajectory_generation_times.append(outcome.generation_time)
            row_routed = row["rollout_expert_indices"]
            if row_routed is not None:
                has_routed_experts = True
                rollout_expert_indices.append(row_routed)
            else:
                rollout_expert_indices.append([])

        successful_sequences.append(list(rows[-1]["prompt_token_ids"]) + list(rows[-1]["response_ids"]))
        successful_rewards.append(outcome.reward)
        successful_times.append(outcome.generation_time)
        successful_turn_counts.append(outcome.num_turns)
        successful_branch_counts.append(len(rows))

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
    rollout_metrics["generate/avg_num_branches"] = (
        sum(successful_branch_counts) / len(successful_branch_counts) if successful_branch_counts else 0.0
    )
    rollout_metrics["generate/num_timeout_trajectories"] = sum(
        1 for outcome in outcomes if outcome.stop_reason == "agent_timeout"
    )
    rollout_metrics["generate/num_error_trajectories"] = sum(
        1 for outcome, rows in zip(outcomes, branch_rows) if outcome.stop_reason == "error" or not rows
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
