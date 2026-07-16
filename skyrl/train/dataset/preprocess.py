import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from jaxtyping import Bool, Float, Integer
from transformers import AutoTokenizer

from skyrl.utils.routed_experts import (
    ROUTED_EXPERT_DTYPES,
    RoutedExpertIndices,
    compact_routed_expert_indices,
)

logger = logging.getLogger(__name__)


def make_router_padding_mask(
    attention_mask: torch.Tensor,
    captured_route_lengths: List[int],
) -> Bool[torch.Tensor, "batch seq_len"]:
    """Build Megatron's router-only padding mask for a ragged vLLM route prefix.

    vLLM records routes only for tokens it evaluates. The final training sequence can be
    longer because the last sampled token has no subsequent decode forward, and SkyRL may
    append a synthetic EOS. In multi-turn generation, observations join the captured prefix
    only when a later turn evaluates them. Captured route rows therefore align with a prefix
    of each real, left-padded sequence; the remaining suffix needs dummy routes.

    This cannot be derived from the loss mask. A loss-masked prompt or observation may still
    condition later trained actions and must replay its captured route. ``True`` marks only
    left padding and tokens without a captured route so Megatron excludes their dummy routes
    from router accounting.
    """
    if attention_mask.ndim != 2:
        raise ValueError(f"Expected 2D attention_mask, got shape {attention_mask.shape}")
    if len(captured_route_lengths) != attention_mask.shape[0]:
        raise ValueError(
            f"Expected one captured route length per trajectory, got {len(captured_route_lengths)} "
            f"for batch size {attention_mask.shape[0]}"
        )

    captured = torch.as_tensor(captured_route_lengths, dtype=torch.long, device=attention_mask.device)
    sequence_lengths = attention_mask.sum(dim=1, dtype=torch.long)
    if torch.any(captured < 0) or torch.any(captured > sequence_lengths):
        raise ValueError(
            f"Captured route lengths must be within trajectory lengths, got "
            f"captured={captured.tolist()} and lengths={sequence_lengths.tolist()}"
        )

    sequence_starts = attention_mask.shape[1] - sequence_lengths
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device).unsqueeze(0)
    captured_positions = (positions >= sequence_starts.unsqueeze(1)) & (
        positions < (sequence_starts + captured).unsqueeze(1)
    )
    return ~captured_positions


def _verify_inputs(
    prompts: List[List[int]],
    responses: List[List[int]],
    rewards: Optional[List[torch.Tensor]],
    loss_masks: List[List[int]],
):
    assert (
        len(prompts) == len(responses) and len(prompts) > 0
    ), "prompts and responses must have the same length and length must be greater than 0, got {} and {}".format(
        len(prompts), len(responses)
    )

    if rewards is not None:
        assert len(rewards) == len(prompts), "rewards must have the same length as prompts, got {} and {}".format(
            len(rewards), len(prompts)
        )
    assert len(loss_masks) == len(prompts), "loss_masks must have the same length as prompt, got {} and {}".format(
        len(loss_masks), len(prompts)
    )


def convert_prompts_responses_to_batch_tensors(
    tokenizer: AutoTokenizer,
    prompts: List[List[int]],
    responses: List[List[int]],
    rewards: List[List[float]],
    loss_masks: List[List[int]],
    logprobs: Optional[List[List[float]]] = None,
    rollout_expert_indices: Optional[List[RoutedExpertIndices]] = None,
    max_seq_len: Optional[int] = None,
) -> Tuple[
    Float[torch.Tensor, "batch seq_len"],
    Float[torch.Tensor, "batch seq_len"],
    Float[torch.Tensor, "batch response_len"],
    Float[torch.Tensor, "batch response_len"],
    Float[torch.Tensor, "batch response_len"],
    Optional[Float[torch.Tensor, "batch response_len"]],
    Optional[Integer[torch.Tensor, "batch seq_len layer_num topk"]],
]:
    """
    Convert prompts and responses to batch tensors for training.

    Each sequence is laid out as a single left-padded block:

    | [PAD]  [PAD]  prompt prompt prompt respon respon |
    | [PAD]  prompt prompt prompt respon respon respon |
    | prompt prompt prompt respon respon respon respon |
                          |<---- max_response_len ---->|

    The padded sequence length is ``max(prompt_len_i + response_len_i)``.
    This way, the max padded sequence length is ``max_seq_len``.

    This makes the response-level tensors (action_mask, rewards, loss_masks, logprobs):
    | prompt prompt respon respon |
    | prompt respon respon respon |
    | respon respon respon respon |

    So the action_mask is:
    | 0       0       1      1    |
    | 0       1       1      1    |
    | 1       1       1      1    |

    Attention mask is 1 for all real tokens, 0 for padding.
    Action mask is 1 for the last ``response_len_i`` positions, 0 for padding.

    Response-level tensors are **right-aligned** within ``(batch, max_response_len)``: non-padded
    values occupy the last ``response_len_i`` positions, with leading zeros. This matches the model
    forward pass which extracts ``log_probs[:, -num_actions-1:-1]`` —- response tokens are always at
    the end of the sequence, so their logprobs are right-aligned in the slice.

    Assumes that the responses already contain an eos token at index -1.

    Args:
        tokenizer: Model tokenizer
        prompts: List of tokenized prompts
        responses: List of tokenized responses
        rewards: List of rewards for each response
        loss_masks: List of loss masks for each response
        logprobs: List of rollout log probs for each response
        max_seq_len: Optional. If provided and ``max(prompt_i + response_i)``
            exceeds it, a warning is logged (no truncation is performed).

    Returns:
        sequences: ``(batch, max_total)`` where ``max_total = max(prompt_i + response_i)``.
        attention_mask: ``(batch, max_total)``
        action_mask: ``(batch, max_response)`` — right-aligned response indicator.
        rewards: ``(batch, max_response)`` — right-aligned.
        loss_masks: ``(batch, max_response)`` — right-aligned.
        logprobs: ``(batch, max_response)`` — right-aligned, or ``None``.
    """
    _verify_inputs(prompts, responses, rewards, loss_masks)

    prompt_token_lens = [len(p) for p in prompts]
    response_token_lens = [len(r) for r in responses]

    max_response = max(response_token_lens)
    # Pad to the tightest bound: max per-sample total.
    max_total = max(p + r for p, r in zip(prompt_token_lens, response_token_lens))

    if max_seq_len is not None and max_total > max_seq_len:
        logger.warning(
            f"Max sequence length in batch ({max_total}) exceeds max_seq_len ({max_seq_len}). "
            f"No truncation is performed; consider checking generator settings."
        )

    pad_token_id = tokenizer.pad_token_id
    sequences = []
    attention_masks = []
    action_masks = []
    for i in range(len(prompts)):
        total_real = prompt_token_lens[i] + response_token_lens[i]
        pad_len = max_total - total_real

        # Unified left-pad: [PAD ... PAD  PROMPT  RESPONSE]
        seq = [pad_token_id] * pad_len + prompts[i] + responses[i]
        attention_mask_i = [0] * pad_len + [1] * total_real

        # Response indicator within the last max_response positions (right-aligned).
        resp_pad = max_response - response_token_lens[i]
        action_mask_i = [0] * resp_pad + [1] * response_token_lens[i]

        sequences.append(seq)
        attention_masks.append(attention_mask_i)
        action_masks.append(action_mask_i)

    sequences = torch.tensor(sequences)
    attention_mask = torch.tensor(attention_masks, dtype=torch.int64)
    action_mask = torch.tensor(action_masks, dtype=torch.int64)

    # Response-level tensors are RIGHT-ALIGNED to match the model output.
    # The model's log_probs[:, -num_actions-1:-1] returns logprobs where
    # response tokens occupy the last response_len_i positions.
    ret_loss_masks = torch.zeros(len(prompts), max_response, dtype=torch.float)
    for i, lm in enumerate(loss_masks):
        ret_loss_masks[i, max_response - len(lm) :] = torch.tensor(lm, dtype=torch.float)

    # Same thing for rewards.
    ret_rewards = torch.zeros(len(prompts), max_response, dtype=torch.float)
    for i, custom_reward in enumerate(rewards):
        if isinstance(custom_reward, list):
            custom_reward = torch.tensor(custom_reward)
        ret_rewards[i, max_response - len(custom_reward) :] = custom_reward

    # Same thing for logprobs.
    logprobs_tensor = None
    if logprobs:
        logprobs_tensor = torch.zeros(len(prompts), max_response, dtype=torch.float)
        for i, sample_logprobs in enumerate(logprobs):
            lp = torch.tensor(sample_logprobs, dtype=torch.float)
            logprobs_tensor[i, max_response - len(sample_logprobs) :] = lp

    rollout_expert_indices_tensor = None
    if rollout_expert_indices is not None:
        num_samples = len(prompts)
        if not isinstance(rollout_expert_indices, list):
            raise TypeError("rollout_expert_indices must be a list of NumPy arrays")
        if len(rollout_expert_indices) != num_samples:
            raise ValueError("rollout_expert_indices must contain routes for every trajectory")

        canonical_indices = []
        for sample_index, sample_indices in enumerate(rollout_expert_indices):
            if not isinstance(sample_indices, np.ndarray):
                raise TypeError(
                    f"rollout_expert_indices entries must be NumPy arrays, got {type(sample_indices).__name__} "
                    f"at sample {sample_index}"
                )
            if sample_indices.dtype not in ROUTED_EXPERT_DTYPES:
                raise TypeError(
                    f"Unsupported routed expert dtype {sample_indices.dtype} at sample {sample_index}; "
                    "expected uint8, int16, or int32"
                )
            canonical_indices.append(compact_routed_expert_indices(sample_indices))

        first_shape = canonical_indices[0].shape
        if len(first_shape) != 3 or first_shape[0] == 0:
            raise ValueError("rollout_expert_indices must contain routes for every trajectory")
        num_layers, topk = first_shape[1:]
        if topk < 1:
            raise ValueError("rollout_expert_indices must contain at least one expert per layer")

        batch_dtype = max((indices.dtype for indices in canonical_indices), key=lambda dtype: dtype.itemsize)
        padded = np.empty((num_samples, max_total, num_layers, topk), dtype=batch_dtype)
        padded[...] = np.arange(topk, dtype=batch_dtype)
        for sample_index, sample_indices in enumerate(canonical_indices):
            if sample_indices.ndim != 3 or sample_indices.shape[1:] != (num_layers, topk):
                raise ValueError(
                    "rollout_expert_indices entries must share [layers, topk], "
                    f"got shape {sample_indices.shape} at sample {sample_index}"
                )
            left_pad = max_total - (prompt_token_lens[sample_index] + response_token_lens[sample_index])
            available = max_total - left_pad
            if sample_indices.shape[0] == 0 or sample_indices.shape[0] > available:
                raise ValueError(
                    f"Trajectory {sample_index} has {sample_indices.shape[0]} route rows for {available} tokens"
                )
            route_end = left_pad + sample_indices.shape[0]
            padded[sample_index, left_pad:route_end] = sample_indices
        rollout_expert_indices_tensor = torch.from_numpy(padded)

    return (
        sequences,
        attention_mask,
        action_mask,
        ret_rewards,
        ret_loss_masks,
        logprobs_tensor,
        rollout_expert_indices_tensor,
    )


def build_dense_sample_support(
    rollout_sample_support: Optional[List[List[List[int]]]],
    response_ids: List[List[int]],
    loss_masks: List[List[int]],
    sequence_length: int,
    top_k: int,
    eos_token_id: int,
) -> Optional[Integer[torch.Tensor, "batch seq_len topk"]]:
    """Validate and left-pad per-token sampler support for replay."""
    if rollout_sample_support is None:
        return None
    if len(rollout_sample_support) != len(response_ids):
        raise ValueError("rollout_sample_support must have one entry per trajectory")
    if len(loss_masks) != len(response_ids):
        raise ValueError("loss_masks must have one entry per trajectory")

    support = torch.full((len(response_ids), sequence_length, top_k), -1, dtype=torch.int32)
    int32_max = int(np.iinfo(np.int32).max)
    for sample_index, (sample_rows, sampled_tokens, sample_loss_mask) in enumerate(
        zip(rollout_sample_support, response_ids, loss_masks, strict=True)
    ):
        if len(sample_rows) != len(sampled_tokens):
            raise ValueError(
                f"rollout_sample_support[{sample_index}] has {len(sample_rows)} rows for "
                f"{len(sampled_tokens)} response tokens"
            )
        if len(sample_loss_mask) != len(sampled_tokens):
            raise ValueError(
                f"loss_masks[{sample_index}] has {len(sample_loss_mask)} entries for "
                f"{len(sampled_tokens)} response tokens"
            )

        sample_support = torch.full((len(sample_rows), top_k), -1, dtype=torch.int64)
        for token_index, row in enumerate(sample_rows):
            if row:
                if len(row) != top_k:
                    raise ValueError("rollout_sample_support rows must match generator.sampling_params.top_k")
                sample_support[token_index] = torch.as_tensor(row, dtype=torch.int64)

        valid = sample_support >= 0
        if torch.any((sample_support < -1) | (sample_support > int32_max)):
            raise ValueError("rollout_sample_support vocab ids must fit non-negative int32")
        if torch.any(valid & ((~valid).cumsum(dim=1) > 0)):
            raise ValueError("rollout_sample_support padding must use trailing -1 values")

        sampled = torch.as_tensor(sampled_tokens, dtype=torch.int64).unsqueeze(1)
        loss_bearing = torch.as_tensor(sample_loss_mask, dtype=torch.bool)
        has_support = valid.any(dim=1)
        unsupported_loss = loss_bearing & ~has_support
        if torch.count_nonzero(unsupported_loss) > 1:
            raise ValueError(f"rollout_sample_support[{sample_index}] has more than one loss-bearing unsupported token")
        unsupported_non_eos = unsupported_loss & (sampled.squeeze(1) != eos_token_id)
        if torch.any(unsupported_non_eos):
            token_index = int(torch.where(unsupported_non_eos)[0][0])
            raise ValueError(
                f"rollout_sample_support[{sample_index}][{token_index}] is empty for a loss-bearing non-EOS token"
            )
        missing = loss_bearing & has_support & ~torch.any(sample_support == sampled, dim=1)
        if torch.any(missing):
            missing_token = sampled_tokens[int(torch.where(missing)[0][0])]
            raise ValueError(f"sampled token {missing_token} is missing from rollout_sample_support")

        start = sequence_length - len(sampled_tokens)
        if start < 0:
            raise ValueError("response tokens exceed the sample-support sequence width")
        support[sample_index, start:] = sample_support.to(torch.int32)
    return support


def compute_prompt_boundaries(uids: List[str]) -> List[Tuple[int, int]]:
    """Compute per-prompt ``(start, end)`` slices from a flat ``uids`` list.

    Args:
        uids: List of uids, representing which prompt each sequence belongs to. Consecutive
            equal entries belong to the same prompt (same assumption as
            ``compute_prompt_mini_batch_boundaries``).

    Returns:
        List of (start, end) indices, one per prompt, in order. Works for both step-wise
        (variable sequences per prompt) and non-step-wise training.

    Example: uids = ["p0", "p0", "p1", "p1", "p1"] -> [(0, 2), (2, 5)]
    """
    boundaries: List[Tuple[int, int]] = []
    seen_uids: set[str] = set()
    start = 0
    for i in range(1, len(uids)):
        if uids[i] != uids[i - 1]:
            assert (
                uids[i] not in seen_uids
            ), f"uid {uids[i]!r} appears in non-contiguous positions at index {i}. Full uids: {uids}"
            seen_uids.add(uids[i - 1])
            boundaries.append((start, i))
            start = i
    if uids:
        boundaries.append((start, len(uids)))
    return boundaries


def compute_prompt_mini_batch_boundaries(
    uids: List[str],
    mini_batch_size: int,
    train_batch_size: int,
    is_stepwise: bool,
    n_samples_per_prompt: int,
) -> List[Tuple[int, int]]:
    """Compute mini-batch ``(start, end)`` slices from a flat ``uids`` list.

    Args:
        uids: List of uids, representing which prompt each sequence belongs to.
        mini_batch_size: Number of prompts to include in each mini-batch. Same as training config's
            config.trainer.policy_mini_batch_size or config.trainer.critic_mini_batch_size.
        train_batch_size: Number of prompts in a training batch. For sanity check.
        is_stepwise: Whether the training is step-wise. For sanity check.
        n_samples_per_prompt: how many samples per prompt. For sanity check.
    Returns:
        List of (start, end) indices of the mini-batches. The length of the list is the number of
        mini-batches, guaranteed to be `train_batch_size // mini_batch_size` regardless of whether
        the training is step-wise or not.

    Consecutive equal entries in ``uids`` belong to the same prompt. Each mini batch spans exactly
    ``mini_batch_size`` prompts (the last may be smaller if the total prompt count is not divisible
    in step-wise training). Works for both step-wise (variable sequences per prompt) and non-step-wise
    (fixed ``n_samples_per_prompt`` sequences per prompt) training.

    We assume uids are contiguous, i.e. all n_samples_per_prompt trajectories for a prompt, or all
    per-step sequences for a trajectory, are contiguous.

    Example A: normal non-step-wise training, with n_samples_per_prompt=2 and train_batch_size=4.
    uids = ["p0", "p0", "p1", "p1", "p2", "p2", "p3", "p3"]
    mini_batch_size = 2
    prompt_end_indices = [2, 4, 6, 8]
    boundaries = [(0, 4), (4, 8)]  # because each mini batch spans exactly 2 prompts, hence 4 sequences

    Example B: step-wise training with n_samples_per_prompt = 2, and each trajectory can have 1-2 turns.
    uids = ["p0", "p0", "p0", "p0", "p1", "p1", "p2", "p2", "p2", "p3", "p3"]
    mini_batch_size = 2
    prompt_end_indices = [4, 6, 9, 11]
    boundaries = [(0, 6), (6, 11)]
    """
    # First compute the end indices of each prompt.
    prompt_end_indices: List[int] = []
    seen_uids: set[str] = set()
    seen_uids.add(uids[0])
    for i in range(1, len(uids)):
        if uids[i] != uids[i - 1]:
            assert (
                uids[i] not in seen_uids
            ), f"uid {uids[i]!r} appears in non-contiguous positions at index {i}. Full uids: {uids}"
            seen_uids.add(uids[i])
            prompt_end_indices.append(i)
    prompt_end_indices.append(len(uids))

    # seen_uids should equal to the number of prompts and equal to `train_batch_size`
    num_prompts = len(prompt_end_indices)
    assert num_prompts == train_batch_size and len(seen_uids) == train_batch_size
    assert train_batch_size % mini_batch_size == 0

    # Compute boundaries.
    boundaries: List[Tuple[int, int]] = []
    start_seq = 0
    for i in range(0, num_prompts, mini_batch_size):
        end_prompt_idx = i + mini_batch_size - 1  # i + mini_batch_size is next mini-batch's first prompt's end index
        end_seq = prompt_end_indices[end_prompt_idx]
        boundaries.append((start_seq, end_seq))
        start_seq = end_seq
    assert len(boundaries) == train_batch_size // mini_batch_size

    # Assert that the mini-batch boundaries are uniform for non-step-wise training.
    if not is_stepwise:
        expected_num_seq_in_mini_batch = n_samples_per_prompt * mini_batch_size
        for i, (start, end) in enumerate(boundaries):
            assert start == i * expected_num_seq_in_mini_batch
            assert end - start == expected_num_seq_in_mini_batch

    return boundaries
