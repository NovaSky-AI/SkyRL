import logging
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float, Integer
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _verify_inputs(
    prompts: List[List[int]],
    responses: List[List[int]],
    rewards: Optional[List[Union[List[float], torch.Tensor]]],
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


def _reward_to_numpy(custom_reward: Union[List[float], torch.Tensor]) -> np.ndarray:
    if isinstance(custom_reward, torch.Tensor):
        reward_arr = custom_reward.detach().to(device="cpu", dtype=torch.float32).numpy()
    else:
        reward_arr = np.asarray(custom_reward, dtype=np.float32)

    if reward_arr.ndim != 1:
        raise ValueError(f"Expected a 1D per-token reward sequence, got shape {reward_arr.shape}")
    return reward_arr


def convert_prompts_responses_to_batch_tensors(
    tokenizer: AutoTokenizer,
    prompts: List[List[int]],
    responses: List[List[int]],
    rewards: List[Union[List[float], torch.Tensor]],
    loss_masks: List[List[int]],
    logprobs: Optional[List[List[float]]] = None,
    rollout_expert_indices: Optional[List[List[List[List[int]]]]] = None,
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
    Convert tokenized prompt/response pairs to left-padded training tensors.

    Sequences use the tightest batch length, ``max(prompt_len + response_len)``.
    Response-level tensors are right-aligned to ``max_response_len`` so they
    match ``log_probs[:, -num_actions-1:-1]`` from the model forward pass.

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
    num_samples = len(prompts)

    # Fill NumPy buffers by slice, then convert once.
    prompt_lens = np.asarray(prompt_token_lens, dtype=np.int64)
    response_lens = np.asarray(response_token_lens, dtype=np.int64)
    total_real = prompt_lens + response_lens  # (num_samples,)
    pad_lens = max_total - total_real  # left-pad width per sample

    # Left-pad each prompt+response row.
    sequences_np = np.full((num_samples, max_total), pad_token_id, dtype=np.int64)
    for i in range(num_samples):
        start = int(pad_lens[i])
        p_len = int(prompt_lens[i])
        sequences_np[i, start : start + p_len] = prompts[i]
        sequences_np[i, start + p_len :] = responses[i]

    # Real tokens occupy the trailing total_real positions.
    col_total = np.arange(max_total, dtype=np.int64)
    attention_mask_np = (col_total[None, :] >= pad_lens[:, None]).astype(np.int64)

    # Response tokens occupy the trailing response_len positions.
    col_resp = np.arange(max_response, dtype=np.int64)
    resp_pad = max_response - response_lens
    action_mask_np = (col_resp[None, :] >= resp_pad[:, None]).astype(np.int64)

    sequences = torch.from_numpy(sequences_np)
    attention_mask = torch.from_numpy(attention_mask_np)
    action_mask = torch.from_numpy(action_mask_np)

    # Response-level tensors are right-aligned to match the model output.
    ret_loss_masks_np = np.zeros((num_samples, max_response), dtype=np.float32)
    for i, lm in enumerate(loss_masks):
        ret_loss_masks_np[i, max_response - len(lm) :] = lm

    # Tensor rewards need explicit CPU/detach handling before NumPy packing.
    ret_rewards_np = np.zeros((num_samples, max_response), dtype=np.float32)
    for i, custom_reward in enumerate(rewards):
        reward_arr = _reward_to_numpy(custom_reward)
        ret_rewards_np[i, max_response - reward_arr.shape[0] :] = reward_arr

    ret_loss_masks = torch.from_numpy(ret_loss_masks_np)
    ret_rewards = torch.from_numpy(ret_rewards_np)

    # Rollout logprobs are right-aligned like rewards and loss masks.
    logprobs_tensor = None
    if logprobs:
        logprobs_np = np.zeros((num_samples, max_response), dtype=np.float32)
        for i, sample_logprobs in enumerate(logprobs):
            logprobs_np[i, max_response - len(sample_logprobs) :] = sample_logprobs
        logprobs_tensor = torch.from_numpy(logprobs_np)

    rollout_expert_indices_tensor = None
    if rollout_expert_indices:
        first_non_empty = next((x for x in rollout_expert_indices if x), None)
        if first_non_empty:
            num_layers = len(first_non_empty[0])
            topk = len(first_non_empty[0][0]) if num_layers > 0 else 0
            padded = torch.zeros(len(rollout_expert_indices), max_total, num_layers, topk, dtype=torch.int32)
            for i, sample_indices in enumerate(rollout_expert_indices):
                if sample_indices:
                    left_pad = max_total - (prompt_token_lens[i] + response_token_lens[i])
                    n = min(len(sample_indices), max_total - left_pad)
                    padded[i, left_pad : left_pad + n] = torch.tensor(sample_indices[:n], dtype=torch.int32)
            rollout_expert_indices_tensor = padded

            # downcast to uint8 if possible, otherwise int16 to save memory
            if rollout_expert_indices_tensor.max().item() < 2**8:
                rollout_expert_indices_tensor = rollout_expert_indices_tensor.to(torch.uint8)
            elif rollout_expert_indices_tensor.max().item() < 2**15:
                rollout_expert_indices_tensor = rollout_expert_indices_tensor.to(torch.int16)

    return (
        sequences,
        attention_mask,
        action_mask,
        ret_rewards,
        ret_loss_masks,
        logprobs_tensor,
        rollout_expert_indices_tensor,
    )


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
