from typing import List, Optional, Sequence, Union


Reward = Union[float, List[float]]


def apply_dapo_soft_overlong_punishment(
    response_ids: List[List[int]],
    rewards: Sequence[Reward],
    overlong_buffer_len: int,
    overlong_buffer_penalty_factor: float,
    *,
    max_response_length: Optional[int] = None,
    max_response_lengths: Optional[Sequence[int]] = None,
) -> List[Reward]:
    """Apply DAPO's soft overlong punishment to response- or token-level rewards.

    Exactly one of ``max_response_length`` or ``max_response_lengths`` must be provided.
    ``max_response_lengths`` exists for prompt-aware integrations that compute a per-sample
    remaining budget instead of using one global scalar.
    """
    if (max_response_length is None) == (max_response_lengths is None):
        raise ValueError("Provide exactly one of `max_response_length` or `max_response_lengths`")
    if overlong_buffer_len <= 0:
        raise ValueError(f"`overlong_buffer_len` must be > 0, got {overlong_buffer_len}")
    if overlong_buffer_penalty_factor < 0:
        raise ValueError(
            "`overlong_buffer_penalty_factor` must be >= 0, "
            f"got {overlong_buffer_penalty_factor}"
        )
    if max_response_length is not None and max_response_length <= 0:
        raise ValueError(f"`max_response_length` must be > 0, got {max_response_length}")
    if len(response_ids) != len(rewards):
        raise ValueError(
            "`response_ids` and `rewards` must have the same length, "
            f"got {len(response_ids)} and {len(rewards)}"
        )

    if max_response_lengths is None:
        max_response_lengths = [max_response_length] * len(response_ids)
    elif len(max_response_lengths) != len(response_ids):
        raise ValueError(
            "`max_response_lengths` and `response_ids` must have the same length, "
            f"got {len(max_response_lengths)} and {len(response_ids)}"
        )

    adjusted_rewards: List[Reward] = []
    for response, reward, sample_max_response_length in zip(response_ids, rewards, max_response_lengths):
        if sample_max_response_length <= 0:
            raise ValueError(
                "`sample_max_response_length` must be > 0, "
                f"got {sample_max_response_length}"
            )
        if overlong_buffer_len > sample_max_response_length:
            raise ValueError(
                "`overlong_buffer_len` must be <= `sample_max_response_length`, "
                f"got {overlong_buffer_len} > {sample_max_response_length}"
            )
        response_length = len(response)
        max_exceed_length = sample_max_response_length - overlong_buffer_len

        if isinstance(reward, list):
            updated_reward: Reward = reward.copy()
        else:
            updated_reward = float(reward)

        if response_length > max_exceed_length and response_length <= sample_max_response_length:
            exceed_length = response_length - max_exceed_length
            penalty = exceed_length / overlong_buffer_len * overlong_buffer_penalty_factor

            if isinstance(updated_reward, list):
                if updated_reward:
                    updated_reward[-1] -= penalty
            else:
                updated_reward -= penalty
        elif response_length > sample_max_response_length:
            if isinstance(updated_reward, list):
                updated_reward = [0.0] * len(updated_reward)
            else:
                updated_reward = 0.0

        adjusted_rewards.append(updated_reward)

    return adjusted_rewards
