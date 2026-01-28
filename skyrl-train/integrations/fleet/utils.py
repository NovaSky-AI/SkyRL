"""
Utility functions for Fleet task training with Tinker.

These functions handle sequence truncation and loss mask filtering,
matching SkyRL's skyrl_gym_generator patterns.
"""

from typing import List, Tuple


def truncate_sequence(
    prompt_ids: List[int],
    response_ids: List[int],
    max_sequence_length: int,
) -> Tuple[List[int], List[int], int]:
    """
    Truncate a sequence to fit within max_sequence_length.

    The prompt is preserved fully; only the response is truncated.

    Args:
        prompt_ids: Token IDs for the prompt.
        response_ids: Token IDs for the response.
        max_sequence_length: Maximum total sequence length.

    Returns:
        Tuple of (full_sequence, truncated_response_ids, response_len).
    """
    full_sequence = prompt_ids + response_ids
    prompt_len = len(prompt_ids)

    if len(full_sequence) > max_sequence_length:
        full_sequence = full_sequence[:max_sequence_length]
        response_len = len(full_sequence) - prompt_len
        truncated_response_ids = response_ids[:response_len]
    else:
        response_len = len(response_ids)
        truncated_response_ids = response_ids

    return full_sequence, truncated_response_ids, response_len


def truncate_auxiliary_data(
    data: List,
    response_len: int,
) -> List:
    """
    Truncate auxiliary data (logprobs, loss_mask) to match truncated response length.

    Args:
        data: List of values corresponding to response tokens.
        response_len: Target length after truncation.

    Returns:
        Truncated list.
    """
    if len(data) > response_len:
        return data[:response_len]
    return data


def apply_overlong_filtering_simple(
    loss_masks: List[List[int]],
    response_ids: List[List[int]],
    eos_token_id: int,
) -> List[List[int]]:
    """
    Apply DAPO overlong filtering: zero out loss mask for responses not ending with EOS.

    This is a simplified version for testing - the actual SkyRL function is in
    skyrl_train.generators.utils.apply_overlong_filtering.

    Args:
        loss_masks: List of loss masks for each response.
        response_ids: List of response token IDs for each response.
        eos_token_id: The EOS token ID.

    Returns:
        Filtered loss masks (zeroed if response doesn't end with EOS).
    """
    filtered = []
    for mask, response in zip(loss_masks, response_ids):
        # Empty response or doesn't end with EOS -> zero out mask
        if not response or response[-1] != eos_token_id:
            filtered.append([0] * len(mask))
        else:
            filtered.append(list(mask))
    return filtered


def prepare_training_sequence(
    prompt_ids: List[int],
    response_ids: List[int],
    logprobs: List[float],
    loss_mask: List[int],
    max_sequence_length: int,
) -> Tuple[List[int], List[float], List[int], bool]:
    """
    Prepare a training sequence with truncation if needed.

    Args:
        prompt_ids: Token IDs for the prompt.
        response_ids: Token IDs for the response.
        logprobs: Log probabilities for response tokens.
        loss_mask: Loss mask for response tokens.
        max_sequence_length: Maximum total sequence length.

    Returns:
        Tuple of (full_sequence, truncated_logprobs, truncated_loss_mask, was_truncated).
    """
    full_sequence, truncated_response, response_len = truncate_sequence(prompt_ids, response_ids, max_sequence_length)

    was_truncated = len(prompt_ids) + len(response_ids) > max_sequence_length

    truncated_logprobs = truncate_auxiliary_data(logprobs, response_len)
    truncated_loss_mask = truncate_auxiliary_data(loss_mask, response_len)

    return full_sequence, truncated_logprobs, truncated_loss_mask, was_truncated
