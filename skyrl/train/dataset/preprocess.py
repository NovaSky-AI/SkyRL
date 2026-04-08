import logging
from typing import List, Optional, Tuple

import torch
from jaxtyping import Float, Integer
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


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
