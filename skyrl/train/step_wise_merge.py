"""
Prefix-aware merging of step-wise trajectory turns for training.

When step_wise_trajectories=True, each turn is initially a separate sample.
We merge consecutive turns into fewer samples only when the next turn's prompt
token IDs have the previous full sequence (prompt + response) as an exact prefix.
Otherwise we keep them as separate samples (token-id prefix match only).
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple


def _is_prefix(sequence: List[int], candidate: List[int]) -> bool:
    """Check if sequence is a prefix of candidate (exact token-id match)."""
    if len(sequence) > len(candidate):
        return False
    return sequence == candidate[: len(sequence)]


@dataclass
class MergedStepWiseSample:
    """A single training sample after merging one or more step-wise turns."""

    prompt_token_ids: List[int]
    response_ids: List[int]
    rewards: List[float]
    loss_masks: List[int]
    rollout_logprobs: Optional[List[float]] = None
    is_last_step: bool = False


def merge_step_wise_turns_for_trajectory(
    prompt_token_ids: List[List[int]],
    response_ids: List[List[int]],
    rewards: List[List[float]],
    loss_masks: List[List[int]],
    is_last_step: List[bool],
    rollout_logprobs: Optional[List[List[float]]] = None,
) -> Tuple[List[MergedStepWiseSample], int]:
    """
    Merge consecutive turns for a single trajectory when the next observation
    has the previous full sequence (prompt + response) as an exact prefix.

    Args:
        prompt_token_ids: Per-turn prompt (observation) token IDs.
        response_ids: Per-turn response (action) token IDs.
        rewards: Per-turn per-token rewards (list of lists).
        loss_masks: Per-turn loss masks (list of lists).
        is_last_step: Per-turn flag True only on the final turn of the trajectory.
        rollout_logprobs: Optional per-turn rollout logprobs (list of lists).

    Returns:
        (merged_samples, prefix_mismatch_count)
        - merged_samples: List of merged training samples for this trajectory.
        - prefix_mismatch_count: Number of times we did not merge due to prefix mismatch.
    """
    n = len(prompt_token_ids)
    assert n == len(response_ids) == len(rewards) == len(loss_masks) == len(is_last_step)
    if rollout_logprobs is not None:
        assert len(rollout_logprobs) == n

    merged: List[MergedStepWiseSample] = []
    prefix_mismatch_count = 0

    # Full sequence so far (obs + response) for prefix check and for building merged prompt
    full_sequence: List[int] = []
    acc_response_ids: List[int] = []
    acc_rewards: List[float] = []
    acc_loss_masks: List[int] = []
    acc_logprobs: List[float] = []
    acc_is_last_step = False
    last_response_len = 0  # length of last turn's response (for prompt = full_sequence[:-last_response_len])

    def flush() -> None:
        """Emit current accumulated sample and reset."""
        nonlocal full_sequence, acc_response_ids, acc_rewards, acc_loss_masks, acc_logprobs, acc_is_last_step, last_response_len
        if not full_sequence and not acc_response_ids:
            return
        # Merged prompt = full context = everything except the last turn's response
        prompt_len = len(full_sequence) - last_response_len
        prompt_tokens = full_sequence[:prompt_len] if prompt_len > 0 else []
        merged.append(
            MergedStepWiseSample(
                prompt_token_ids=prompt_tokens,
                response_ids=list(acc_response_ids),
                rewards=list(acc_rewards),
                loss_masks=list(acc_loss_masks),
                rollout_logprobs=list(acc_logprobs) if (rollout_logprobs is not None) else None,
                is_last_step=acc_is_last_step,
            )
        )
        full_sequence = []
        acc_response_ids = []
        acc_rewards = []
        acc_loss_masks = []
        acc_logprobs = []
        acc_is_last_step = False
        last_response_len = 0

    for i in range(n):
        ob_tokens = prompt_token_ids[i]
        ac_tokens = response_ids[i]
        ac_rewards = rewards[i]
        ac_masks = loss_masks[i]
        ac_logprobs_i = rollout_logprobs[i] if rollout_logprobs is not None else [0.0] * len(ac_tokens)

        if len(full_sequence) == 0:
            delta_ob = ob_tokens
        elif _is_prefix(full_sequence, ob_tokens):
            delta_ob = ob_tokens[len(full_sequence) :]
        else:
            prefix_mismatch_count += 1
            flush()
            delta_ob = ob_tokens

        full_sequence.extend(delta_ob)
        full_sequence.extend(ac_tokens)
        acc_response_ids.extend(ac_tokens)
        acc_rewards.extend(ac_rewards)
        acc_loss_masks.extend(ac_masks)
        acc_logprobs.extend(ac_logprobs_i)
        acc_is_last_step = acc_is_last_step or is_last_step[i]
        last_response_len = len(ac_tokens)

    flush()
    return merged, prefix_mismatch_count
