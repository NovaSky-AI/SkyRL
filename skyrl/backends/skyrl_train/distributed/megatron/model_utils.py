# Utils ported from NeMo-Aligner by way of NeMo-RL
# https://github.com/NVIDIA-NeMo/RL/blob/9301d36cbf847212430b84a27cfe6990f773b7cf/nemo_rl/distributed/model_utils.py#L4
# The original copyright is reproduced below:

#  Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Optional

import megatron.core.parallel_state as mpu
import torch
import torch.distributed as dist


@torch.no_grad()
def _compute_distributed_log_softmax(
    vocab_parallel_logits: torch.Tensor, group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """Compute a stable distributed log softmax across tensor parallel workers.

    Taken from: https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L265

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_length, vocab_size//TP]
            where TP is the tensor parallel size.
        group (torch.distributed.ProcessGroup): Process group for the all-reduce operations.

    Returns:
        torch.Tensor: Log softmax output with the same shape as input, but values represent
            log probabilities normalized across the full vocabulary dimension.
    """
    logits_max = torch.amax(vocab_parallel_logits, dim=-1, keepdim=True)
    torch.distributed.all_reduce(
        logits_max,
        op=torch.distributed.ReduceOp.MAX,
        group=group,
    )

    # Subtract the maximum value.
    vocab_parallel_logits = vocab_parallel_logits - logits_max

    sum_exp_logits = vocab_parallel_logits.exp().sum(-1, keepdim=True).float()

    torch.distributed.all_reduce(
        sum_exp_logits,
        op=torch.distributed.ReduceOp.SUM,
        group=group,
    )

    return vocab_parallel_logits - sum_exp_logits.log_().to(vocab_parallel_logits.dtype)


class DistributedLogprob(torch.autograd.Function):
    """Custom autograd function for computing log probabilities in a distributed setting.

    Taken from https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L286
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]  Always ignore torch.autograd.Function.forward's type since it's always more specific than the base class
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        vocab_parallel_logits = vocab_parallel_logits.to(dtype=torch.float32)

        log_probs = _compute_distributed_log_softmax(vocab_parallel_logits, group=group)
        softmax_output = log_probs.exp()

        log_probs = torch.gather(log_probs, -1, masked_target.unsqueeze(-1)).squeeze(-1)
        log_probs[target_mask] = 0.0

        torch.distributed.all_reduce(
            log_probs,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )

        if not inference_only:
            # only save for backward when we have inference only=False
            ctx.save_for_backward(softmax_output, target_mask, masked_target)

        return log_probs

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        grad_output = grad_outputs[0]
        softmax, target_mask, masked_target = ctx.saved_tensors

        if softmax.ndim == 3:
            B, S, V = softmax.shape

            # skip `torch.nn.functional.one_hot`
            row = torch.arange(B, device=softmax.device).view(-1, 1).expand(-1, S).reshape(-1)
            col = torch.arange(S, device=softmax.device).expand(B, -1).reshape(-1)
            flat_idx = (row * S + col) * V

            flat_chosen = flat_idx.masked_select(~target_mask.reshape(-1)) + masked_target.masked_select(~target_mask)

            # `neg` is zero-copy
            grad_input = softmax.neg()
            grad_input = grad_input.mul_(grad_output.unsqueeze(-1))

            grad_output_selected = grad_output.masked_select(~target_mask)
            grad_input.view(-1).scatter_add_(0, flat_chosen, grad_output_selected)
        else:
            V = softmax.size(-1)
            is_chosen = (~target_mask).unsqueeze(-1) * torch.nn.functional.one_hot(masked_target, num_classes=V)
            grad_input = is_chosen.float().sub_(softmax)
            grad_input.mul_(grad_output.unsqueeze(-1))

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None, None, None, None, None, None


class ChunkedDistributedLogprob(torch.autograd.Function):
    """Custom autograd function for computing log probabilities in a distributed setting.

    The log probabilities computation is chunked in the sequence dimension
    to mitigate GPU OOM (especially during backward pass).
    In addition, logits casting from float16 or bfloat16 -> float32 is performed
    inside the chunk loop to avoid materializing a whole float32 logits tensor.

    Adapted from https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L286
    """

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]  Always ignore torch.autograd.Function.forward's type since it's always more specific than the base class
        ctx: Any,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        vocab_start_index: int,
        vocab_end_index: int,
        chunk_size: int,
        tp_group: torch.distributed.ProcessGroup,
        inference_only: bool = False,
    ) -> torch.Tensor:
        # Create a mask of valid vocab ids (1 means it needs to be masked).
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = target - vocab_start_index
        masked_target[target_mask] = 0

        seq_size = int(vocab_parallel_logits.shape[1])
        num_chunks = (seq_size + chunk_size - 1) // chunk_size
        all_log_probs = []

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(seq_size, (chunk_idx + 1) * chunk_size)

            logits = vocab_parallel_logits[:, chunk_start:chunk_end, :]
            logits = logits.to(dtype=torch.float32)

            log_probs = _compute_distributed_log_softmax(
                logits,
                group=tp_group,
            )

            log_probs = torch.gather(log_probs, -1, masked_target[:, chunk_start:chunk_end].unsqueeze(-1)).squeeze(-1)
            log_probs[target_mask[:, chunk_start:chunk_end]] = 0.0

            torch.distributed.all_reduce(
                log_probs,
                op=torch.distributed.ReduceOp.SUM,
                group=tp_group,
            )

            all_log_probs.append(log_probs)

        log_probs = torch.cat(all_log_probs, dim=1)

        if not inference_only:
            # only save for backward when we have inference only=False
            ctx.save_for_backward(vocab_parallel_logits, target_mask, masked_target)
            ctx.chunk_size = chunk_size
            ctx.tp_group = tp_group

        return log_probs

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None, None]:
        grad_output = grad_outputs[0]
        vocab_parallel_logits, target_mask, masked_target = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        tp_group = ctx.tp_group

        partition_vocab_size = int(vocab_parallel_logits.shape[-1])
        seq_size = int(vocab_parallel_logits.shape[1])
        num_chunks = (seq_size + chunk_size - 1) // chunk_size

        all_grad_input = []

        batch_size = int(vocab_parallel_logits.shape[0])

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(seq_size, (chunk_idx + 1) * chunk_size)
            chunk_len = chunk_end - chunk_start

            logits = vocab_parallel_logits[:, chunk_start:chunk_end, :]
            logits = logits.to(dtype=torch.float32)

            softmax_output = _compute_distributed_log_softmax(
                logits,
                group=tp_group,
            )
            softmax_output = softmax_output.exp()

            # Memory-efficient scatter-add fast path (ported from DistributedLogprob.backward).
            # Materializing one_hot(masked_target, num_classes=partition_vocab_size) would
            # allocate a [B, chunk_len, partition_vocab_size] int64 tensor (~8x the size of
            # softmax_output in float32), which causes OOM for large vocabularies. Instead,
            # compute -softmax * grad_output in place and add grad_output at the chosen-token
            # positions via scatter_add_.
            chunk_target_mask = target_mask[:, chunk_start:chunk_end]
            chunk_masked_target = masked_target[:, chunk_start:chunk_end]
            chunk_grad_output = grad_output[:, chunk_start:chunk_end]

            row = torch.arange(batch_size, device=softmax_output.device).view(-1, 1).expand(-1, chunk_len).reshape(-1)
            col = torch.arange(chunk_len, device=softmax_output.device).expand(batch_size, -1).reshape(-1)
            # Flat offset to the start of each [b, s, :] row in the chunk's flattened tensor.
            flat_idx = (row * chunk_len + col) * partition_vocab_size

            valid_mask = ~chunk_target_mask
            flat_chosen = flat_idx.masked_select(valid_mask.reshape(-1)) + chunk_masked_target.masked_select(valid_mask)

            # `neg` is zero-copy; the subsequent mul_ writes in place.
            grad_input = softmax_output.neg_()
            grad_input.mul_(chunk_grad_output.unsqueeze(-1))

            grad_output_selected = chunk_grad_output.masked_select(valid_mask)
            grad_input.view(-1).scatter_add_(0, flat_chosen, grad_output_selected)

            all_grad_input.append(grad_input)

        grad_input = torch.cat(all_grad_input, dim=1)

        # if you add an argument to the forward method, then you must add a corresponding None here
        return grad_input, None, None, None, None, None, None


def from_parallel_logits_to_logprobs(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Get log probabilities from TP+CP sharded vocab logits.

    Args:
        vocab_parallel_logits (torch.Tensor): Logits tensor with shape [batch_size, seq_len // CP, vocab_size // TP]
            where TP is the tensor parallel size.
        target (torch.Tensor): Target token indices with shape [batch_size, seq_len].
            NOTE: Must be the unmodified targets as this function will shift them internally.
        vocab_start_index (int): Starting vocabulary index for this worker's partition.
        vocab_end_index (int): Ending vocabulary index for this worker's partition.
        tp_group (torch.distributed.ProcessGroup): Process group for distributed communication.
        inference_only (bool, optional): If True, tensors won't be saved for backward pass. Defaults to False.
        cp_group (torch.distributed.ProcessGroup, optional): Context parallelism process group. Defaults to None.
        chunk_size (int, optional): Sequence dimension chunk size for computing the log probabilities.

    Returns:
        torch.Tensor: Log probabilities tensor with shape [batch_size, seq_len-1].
            The sequence dimension is reduced by 1 due to the target shifting.

    Taken from: https://github.com/NVIDIA/NeMo-Aligner/blob/9faab404f21994a7eb1d6ed5890b76152b941636/nemo_aligner/utils/distributed.py#L354
    """
    target = target.roll(shifts=-1, dims=-1)
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    pad_len = 0
    # if cp_size > 1:
    # Pad the targets to local size * cp_size
    pad_len = vocab_parallel_logits.shape[1] * cp_size - target.shape[1]
    if pad_len > 0:
        target = torch.nn.functional.pad(target, (0, pad_len), value=0)

    # Shard the targets by context parallelism
    cp_rank = torch.distributed.get_rank(cp_group)
    target = _get_tokens_on_this_cp_rank(target, cp_rank, cp_size, seq_dim=1)

    # Only use the chunked path when chunking actually splits the sequence into
    # multiple chunks. When chunk_size >= seq_len the whole sequence is one
    # chunk, but ChunkedDistributedLogprob still saves the raw
    # vocab_parallel_logits and recomputes softmax in backward (~3x peak memory
    # vs DistributedLogprob's ~2x), so chunking actively hurts in that regime.
    seq_len_local = vocab_parallel_logits.shape[1]
    if chunk_size is not None and chunk_size < seq_len_local:
        logprobs: torch.Tensor = ChunkedDistributedLogprob.apply(  # type: ignore
            vocab_parallel_logits,
            target,
            vocab_start_index,
            vocab_end_index,
            chunk_size,
            tp_group,
            inference_only,
        ).contiguous()
    else:
        logprobs: torch.Tensor = DistributedLogprob.apply(  # type: ignore
            vocab_parallel_logits,
            target,
            vocab_start_index,
            vocab_end_index,
            tp_group,
            inference_only,
        ).contiguous()

    if cp_size > 1:
        # we need to gather the logits by context parallelism
        logprobs = allgather_cp_sharded_tensor(logprobs, cp_group, seq_dim=1)  # , unpadded_seqlen=target.shape[1])

    if pad_len > 0:
        logprobs = logprobs[:, :-pad_len]

    return logprobs[:, :-1]


def from_parallel_logits_to_logprobs_packed_sequences(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    unpacked_seqlen: int,
    vocab_start_index: int,
    vocab_end_index: int,
    group: torch.distributed.ProcessGroup,
    inference_only: bool = False,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    chunk_size: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Get log probabilities from TP sharded vocab logits for packed sequences.

    Args:
        vocab_parallel_logits (torch.Tensor): Packed logits tensor with shape [1, T // CP, vocab_size//TP]
            where T is the total number of tokens across all packed sequences.
        target (torch.Tensor): Packed target token indices with shape [1, T].
            NOTE: Must be the unmodified targets as this function will shift them internally.
        cu_seqlens (torch.Tensor): Cumulative sequence lengths tensor with shape [batch_size + 1].
            cu_seqlens[i] indicates the start position of sequence i in the packed format.
        unpacked_seqlen (int): The length of the unpacked sequence tensor.
        vocab_start_index (int): Starting vocabulary index for this worker's partition.
        vocab_end_index (int): Ending vocabulary index for this worker's partition.
        group (torch.distributed.ProcessGroup): Process group for distributed communication.
        inference_only (bool, optional): If True, tensors won't be saved for backward pass. Defaults to False.
        cp_group (torch.distributed.ProcessGroup, optional): Context parallelism process group. Defaults to None.
        chunk_size (int, optional): Sequence dimension chunk size for computing the log probabilities.
        attention_mask (torch.Tensor, optional): Original unpacked attention mask with shape [batch_size, unpacked_seqlen].
            When provided, packed log probabilities are scattered back to their original padded sequence positions.

    Returns:
        torch.Tensor: Unpacked log probabilities tensor with shape [batch_size, unpacked_seqlen-1].
            The total length is reduced by batch_size due to target shifting (one token per sequence).
    """
    # Remove batch dimension to work with [T, vocab_size] and [T]
    vocab_parallel_logits = vocab_parallel_logits.squeeze(0)
    target = target.squeeze(0)

    batch_size = cu_seqlens_padded.shape[0] - 1
    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    cp_rank = 0 if cp_group is None else torch.distributed.get_rank(cp_group)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device=target.device, dtype=torch.bool)

    cu_seqlens_padded, _, seq_indices, seq_offsets, seq_lens_padded = _packed_sequence_indices(
        cu_seqlens_padded, target.shape[0], target.device
    )

    next_offsets = torch.remainder(seq_offsets + 1, seq_lens_padded[seq_indices])
    rolled_targets_full = target[cu_seqlens_padded[seq_indices] + next_offsets]
    if cp_size > 1:
        cp_rank_for_token, local_indices = _packed_cp_rank_and_local_indices(
            cu_seqlens_padded, seq_indices, seq_offsets, seq_lens_padded, cp_size
        )
        rolled_targets = torch.empty(target.shape[0] // cp_size, dtype=target.dtype, device=target.device)
        current_rank_mask = cp_rank_for_token == cp_rank
        rolled_targets[local_indices[current_rank_mask]] = rolled_targets_full[current_rank_mask]
    else:
        rolled_targets = rolled_targets_full

    # Add batch dimension back for DistributedLogprob
    rolled_targets = rolled_targets.unsqueeze(0)
    vocab_parallel_logits = vocab_parallel_logits.unsqueeze(0)

    # Apply distributed log probability computation.
    #
    # Only use the chunked path when chunking actually splits the sequence into
    # multiple chunks. When chunk_size >= seq_len the whole sequence is one
    # chunk, but ChunkedDistributedLogprob still saves the raw
    # vocab_parallel_logits and recomputes softmax in backward (~3x peak memory
    # vs DistributedLogprob's ~2x), so chunking actively hurts in that regime.
    seq_len_local = vocab_parallel_logits.shape[1]
    if chunk_size is not None and chunk_size < seq_len_local:
        probs: torch.Tensor = ChunkedDistributedLogprob.apply(  # type: ignore
            vocab_parallel_logits,
            rolled_targets,
            vocab_start_index,
            vocab_end_index,
            chunk_size,
            group,
            inference_only,
        ).contiguous()
    else:
        probs: torch.Tensor = DistributedLogprob.apply(  # type: ignore
            vocab_parallel_logits,
            rolled_targets,
            vocab_start_index,
            vocab_end_index,
            group,
            inference_only,
        ).contiguous()

    # Remove batch dimension for filtering
    probs = probs.squeeze(0)

    # Ensure probs is 1D after squeezing
    if probs.dim() != 1:
        raise ValueError(
            f"Expected probs to be 1D after squeezing, but got shape {probs.shape}. "
            f"Original shape before squeeze: {probs.unsqueeze(0).shape}"
        )

    if cp_size > 1:
        probs = allgather_cp_sharded_packed_tensor(probs, cu_seqlens_padded, cp_group)

    out_logprobs = torch.zeros((batch_size, unpacked_seqlen - 1), dtype=probs.dtype, device=probs.device)
    _, _, seq_indices, seq_offsets, seq_lens_padded = _packed_sequence_indices(
        cu_seqlens_padded, probs.shape[0], probs.device
    )

    if attention_mask is not None:
        seq_lens = attention_mask.sum(dim=1, dtype=torch.long)
        token_ordinals = attention_mask.to(torch.long).cumsum(dim=1)
        output_mask = attention_mask[:, :-1] & (token_ordinals[:, :-1] < seq_lens.unsqueeze(1))
        valid_counts = torch.clamp(seq_lens - 1, min=0)
        packed_mask = seq_offsets < valid_counts[seq_indices]
        out_logprobs[output_mask] = probs[packed_mask]
        return out_logprobs

    valid_counts = torch.clamp(seq_lens_padded - 1, min=0)
    packed_mask = (seq_offsets < valid_counts[seq_indices]) & (seq_offsets < unpacked_seqlen - 1)
    out_logprobs[seq_indices[packed_mask], seq_offsets[packed_mask]] = probs[packed_mask]

    return out_logprobs


def _packed_sequence_indices(
    cu_seqlens_padded: torch.Tensor, total_tokens: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cu_seqlens_padded = cu_seqlens_padded.to(device=device, dtype=torch.long)
    token_indices = torch.arange(total_tokens, device=device)
    seq_indices = torch.searchsorted(cu_seqlens_padded[1:], token_indices, right=True)
    seq_offsets = token_indices - cu_seqlens_padded[seq_indices]
    seq_lens_padded = cu_seqlens_padded[1:] - cu_seqlens_padded[:-1]
    return cu_seqlens_padded, token_indices, seq_indices, seq_offsets, seq_lens_padded


def _packed_cp_rank_and_local_indices(
    cu_seqlens_padded: torch.Tensor,
    seq_indices: torch.Tensor,
    seq_offsets: torch.Tensor,
    seq_lens_padded: torch.Tensor,
    cp_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cp_size == 1:
        return torch.zeros_like(seq_indices), cu_seqlens_padded[seq_indices] + seq_offsets

    seq_lens_for_token = seq_lens_padded[seq_indices]
    chunk_size = torch.div(seq_lens_for_token, 2 * cp_size, rounding_mode="floor")
    chunk_indices = torch.div(seq_offsets, chunk_size, rounding_mode="floor")
    rank_for_token = torch.where(chunk_indices < cp_size, chunk_indices, 2 * cp_size - chunk_indices - 1)
    within_chunk_offsets = seq_offsets - chunk_indices * chunk_size
    within_rank_offsets = torch.where(
        chunk_indices < cp_size,
        within_chunk_offsets,
        chunk_size + within_chunk_offsets,
    )
    local_starts = torch.div(cu_seqlens_padded[:-1], cp_size, rounding_mode="floor")
    local_indices = local_starts[seq_indices] + within_rank_offsets
    return rank_for_token, local_indices


def _get_tokens_on_this_cp_rank(
    input_ids: torch.Tensor,
    cp_rank: int,
    cp_size: int,
    seq_dim: int = 1,
) -> torch.Tensor:
    """Get tokens on this context parallelism rank.

    Assumes that input_ids are already padded to a multiple of cp_size * 2 or cp_size == 1.

    Args:
        input_ids: Input token IDs [seq_length, ]
        cp_rank: Context parallelism rank
        cp_size: Context parallelism size

    Returns:
        Tokens on this context parallelism rank [1, seq_length // cp_size]
    """
    if cp_size == 1:
        return input_ids

    # load balance for causal attention
    shard_size = input_ids.shape[seq_dim] // (cp_size * 2)
    shard_inds = (cp_rank, (cp_size * 2) - cp_rank - 1)

    # Create slices for each dimension
    slices = [slice(None)] * input_ids.dim()
    ids_chunks = []

    for ind in shard_inds:
        slices[seq_dim] = slice(ind * shard_size, (ind + 1) * shard_size)
        ids_chunks.append(input_ids[slices])

    ids = torch.cat(ids_chunks, dim=seq_dim)
    return ids


def allgather_cp_sharded_tensor(tensor, cp_group, seq_dim=1):  # , unpadded_seqlen=None):
    return AllGatherCPTensor.apply(tensor, cp_group, seq_dim)  # , unpadded_seqlen)


def allgather_cp_sharded_packed_tensor(tensor, cu_seqlens_padded, cp_group):
    return AllGatherPackedCPTensor.apply(tensor, cu_seqlens_padded, cp_group)


def vocab_parallel_entropy_packed_sequences(
    vocab_parallel_logits: torch.Tensor,
    cu_seqlens_padded: torch.Tensor,
    unpacked_seqlen: int,
    num_actions: int,
    attention_mask: torch.Tensor,
    loss_mask: Optional[torch.Tensor],
    cp_group: Optional[torch.distributed.ProcessGroup],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute action-token entropy directly on TP+CP sharded packed logits.

    Returns:
        A tuple of (global entropy metric, local entropy term for loss). The
        local term is normalized by the global action-token count. Megatron DDP
        later averages gradients across CP ranks, so callers should keep the
        existing CP-size correction on the entropy loss term.
    """
    entropy_tokens = vocab_parallel_entropy(vocab_parallel_logits).squeeze(0)
    device = entropy_tokens.device
    dtype = entropy_tokens.dtype

    attention_mask = attention_mask.to(device=device, dtype=torch.bool)
    cu_seqlens_padded = cu_seqlens_padded.to(device=device, dtype=torch.long)
    batch_size = attention_mask.shape[0]

    action_weights = torch.zeros((batch_size, unpacked_seqlen - 1), dtype=dtype, device=device)
    if loss_mask is None:
        action_weights[:, -num_actions:] = 1.0
    else:
        action_weights[:, -num_actions:] = loss_mask.to(device=device, dtype=dtype)

    seq_lens = attention_mask.sum(dim=1, dtype=torch.long)
    token_ordinals = attention_mask.to(torch.long).cumsum(dim=1)
    output_mask = attention_mask[:, :-1] & (token_ordinals[:, :-1] < seq_lens.unsqueeze(1))

    token_offsets = token_ordinals - 1
    packed_indices = cu_seqlens_padded[:-1].unsqueeze(1) + token_offsets
    packed_weights = torch.zeros((cu_seqlens_padded[-1],), dtype=dtype, device=device)
    packed_weights[packed_indices[:, :-1][output_mask]] = action_weights[output_mask]

    cp_size = 1 if cp_group is None else torch.distributed.get_world_size(cp_group)
    if cp_size > 1:
        cp_rank = torch.distributed.get_rank(cp_group)
        _, _, seq_indices, seq_offsets, seq_lens_padded = _packed_sequence_indices(
            cu_seqlens_padded, packed_weights.shape[0], device
        )
        cp_rank_for_token, local_indices = _packed_cp_rank_and_local_indices(
            cu_seqlens_padded, seq_indices, seq_offsets, seq_lens_padded, cp_size
        )
        local_weights = torch.zeros_like(entropy_tokens)
        current_rank_mask = cp_rank_for_token == cp_rank
        local_weights[local_indices[current_rank_mask]] = packed_weights[current_rank_mask]
    else:
        local_weights = packed_weights

    local_entropy_sum = (entropy_tokens * local_weights).sum()
    local_count = local_weights.sum()
    global_count = local_count.detach().clone()
    global_entropy_sum = local_entropy_sum.detach().clone()
    if cp_size > 1:
        torch.distributed.all_reduce(global_count, group=cp_group)
        torch.distributed.all_reduce(global_entropy_sum, group=cp_group)
    global_count = global_count.clamp(min=1.0)

    entropy = global_entropy_sum / global_count
    entropy_for_loss = local_entropy_sum / global_count
    return entropy, entropy_for_loss


class AllGatherPackedCPTensor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor, cu_seqlens_padded: torch.Tensor, cp_group: torch.distributed.ProcessGroup):
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank_chunks = [torch.empty_like(tensor) for _ in range(cp_size)]
        torch.distributed.all_gather(tensor_list=cp_rank_chunks, tensor=tensor, group=cp_group)

        total_tokens = tensor.shape[0] * cp_size
        cu_seqlens_padded, _, seq_indices, seq_offsets, seq_lens_padded = _packed_sequence_indices(
            cu_seqlens_padded, total_tokens, tensor.device
        )
        cp_rank_for_token, local_indices = _packed_cp_rank_and_local_indices(
            cu_seqlens_padded, seq_indices, seq_offsets, seq_lens_padded, cp_size
        )

        gathered = torch.stack(cp_rank_chunks, dim=0)
        output = gathered[cp_rank_for_token, local_indices]

        ctx.cp_group = cp_group
        ctx.save_for_backward(cu_seqlens_padded)
        ctx.local_tokens = tensor.shape[0]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        cp_size = torch.distributed.get_world_size(ctx.cp_group)
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        (cu_seqlens_padded,) = ctx.saved_tensors

        torch.distributed.all_reduce(grad_output, group=ctx.cp_group)

        cu_seqlens_padded, _, seq_indices, seq_offsets, seq_lens_padded = _packed_sequence_indices(
            cu_seqlens_padded, grad_output.shape[0], grad_output.device
        )
        cp_rank_for_token, local_indices = _packed_cp_rank_and_local_indices(
            cu_seqlens_padded, seq_indices, seq_offsets, seq_lens_padded, cp_size
        )

        local_rank_mask = cp_rank_for_token == cp_rank
        grad_input = torch.zeros(ctx.local_tokens, dtype=grad_output.dtype, device=grad_output.device)
        grad_input[local_indices[local_rank_mask]] = grad_output[local_rank_mask]
        return grad_input, None, None


class AllGatherCPTensor(torch.autograd.Function):
    def forward(
        ctx, tensor, cp_group: torch.distributed.ProcessGroup, seq_dim=1
    ):  # , unpadded_seqlen: Optional[int] = None):
        cp_size = torch.distributed.get_world_size(cp_group)
        cp_rank_chunks = []
        for _ in range(cp_size):
            cp_rank_chunks.append(torch.empty_like(tensor))

        torch.distributed.all_gather(tensor_list=cp_rank_chunks, tensor=tensor, group=cp_group)

        # undo the CP load balancing chunking
        tensor_chunks = []
        for logit_chunk in cp_rank_chunks:
            tensor_chunks.extend(torch.chunk(logit_chunk, chunks=2, dim=seq_dim))

        chunk_indices = []
        for cp_rank in range(cp_size):
            chunk_indices.append(cp_rank)
            chunk_indices.append(2 * cp_size - cp_rank - 1)

        chunks_and_indices = list(zip(tensor_chunks, chunk_indices))
        chunks_and_indices = sorted(chunks_and_indices, key=lambda tup: tup[1])
        ret_tensor = [chunk for chunk, _ in chunks_and_indices]
        ret_tensor = torch.cat(ret_tensor, dim=seq_dim)

        ctx.seq_dim = seq_dim
        ctx.cp_group = cp_group
        # ctx.unpadded_seqlen = unpadded_seqlen

        return ret_tensor

    def backward(ctx, grad_output):
        cp_size = torch.distributed.get_world_size(ctx.cp_group)
        cp_rank = torch.distributed.get_rank(ctx.cp_group)
        torch.distributed.all_reduce(grad_output, group=ctx.cp_group)

        # chunk the seqdim in 2*cp chunks, and select with a CP load balanced indexing
        seq_dim = ctx.seq_dim
        # if ctx.unpadded_seqlen is not None:
        # # Zero out grad_output along the seq_dim after unpadded_seqlen
        # slicer = [slice(None)] * grad_output.dim()
        # slicer[seq_dim] = slice(ctx.unpadded_seqlen, None)
        #     grad_output[tuple(slicer)] = 0

        grad_output = grad_output.view(
            *grad_output.shape[0:seq_dim],
            2 * cp_size,
            grad_output.shape[seq_dim] // (2 * cp_size),
            *grad_output.shape[(seq_dim + 1) :],
        )

        index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
            non_blocking=True
        )

        grad_input = grad_output.index_select(seq_dim, index)
        grad_input = grad_input.view(*grad_input.shape[0:seq_dim], -1, *grad_input.shape[(seq_dim + 2) :])

        return grad_input, None, None  # , None


# Below ported from https://github.com/volcengine/verl/blob/main/verl/utils/megatron/tensor_parallel.py#L109
class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
        @torch.compile(dynamic=True)
        def mul_reduce(a, b):
            return (a * b).sum(dim=-1, keepdim=True)

        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=mpu.get_tensor_model_parallel_group())
        normalized_vocab_parallel_logits = vocab_parallel_logits - logits_max
        normalized_exp_logits = normalized_vocab_parallel_logits.exp_()
        normalized_sum_exp_logits = normalized_exp_logits.sum(dim=-1, keepdim=True)
        dist.all_reduce(normalized_sum_exp_logits, group=mpu.get_tensor_model_parallel_group())
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        sum_softmax_times_logits = mul_reduce(softmax_logits, vocab_parallel_logits)
        dist.all_reduce(sum_softmax_times_logits, group=mpu.get_tensor_model_parallel_group())
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        ctx.save_for_backward(vocab_parallel_logits, softmax_logits, sum_softmax_times_logits)
        return entropy.squeeze(dim=-1)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        vocab_parallel_logits, softmax_logits, sum_softmax_times_logits = ctx.saved_tensors
        # grad = softmax * (sum_softmax_times_logits - vocab_parallel_logits) * grad_output
        # NOTE: do NOT mutate vocab_parallel_logits in-place. The same logits tensor may also
        # be saved for backward by ChunkedDistributedLogprob; even the "sub_ then add_" restore
        # pattern bumps the storage version counter and trips that Function's version check.
        softmax_logits.mul_(sum_softmax_times_logits - vocab_parallel_logits)
        softmax_logits.mul_(grad_output.unsqueeze(dim=-1))
        return softmax_logits


def vocab_parallel_entropy(vocab_parallel_logits: torch.Tensor) -> torch.Tensor:
    """Compute entropy when the logits are sharded in tp ranks

    Args:
        vocab_parallel_logits: (total_nnz, vocab_size // tp_size)

    Returns: (total_nnz,)

    """
    return _VocabParallelEntropy.apply(vocab_parallel_logits)
