"""Score centering on top of the sample-support channel.

Score centering (https://arxiv.org/abs/2609.20807) needs, for every loss-active response token, the
sampler's head (its most probable next tokens with their sampler logprobs) and the trainer's logprob of
each head member. The sampler head arrives as two row-aligned packed side channels,
``rollout_sample_support`` (ids) and ``rollout_sample_support_logprobs``. This module aligns those rows
to response positions and scores the head members under the trainer without materializing the
vocabulary logits for them, on unsharded (FSDP) and tensor-parallel (Megatron) vocabularies alike.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
    align_token_metadata,
    canonical_token_metadata_layout,
)
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_LOGPROBS_PADDING,
    SAMPLE_SUPPORT_PADDING,
    align_sample_support_row_ids,
)
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    _selected_hidden_projection,
    sample_support_row_ids_in_batch_positions,
)
from skyrl.backends.skyrl_train.utils.torch_utils import masked_mean


def gather_packed_rows(packed: PackedTensor, row_ids: torch.Tensor, padding_value: float | int) -> torch.Tensor:
    """Gather one ``[*row_shape]`` row per position by packed row id; negative ids yield ``padding_value``."""
    row_shape = tuple(packed.values.shape[1:])
    if packed.values.shape[0] == 0:
        return torch.full((*row_ids.shape, *row_shape), padding_value, dtype=packed.dtype, device=packed.device)
    flat_row_ids = row_ids.reshape(-1)
    gathered = packed.values.index_select(0, flat_row_ids.clamp(min=0))
    invalid = (flat_row_ids < 0).reshape(-1, *([1] * len(row_shape)))
    gathered = gathered.masked_fill(invalid, padding_value)
    return gathered.reshape(*row_ids.shape, *row_shape)


@dataclass(frozen=True)
class SamplerHead:
    """The sampler's recorded head at every response position, in ``[batch, num_actions, k]`` layout."""

    ids: torch.Tensor
    """Vocabulary ids, ``SAMPLE_SUPPORT_PADDING`` (-1) where a row holds fewer than ``k`` members."""
    logprobs: torch.Tensor
    """Sampler logprobs of ``ids`` (float32), ``-inf`` on padding entries."""


def sampler_head_for_actions(
    sample_support: PackedTensor,
    sample_support_logprobs: PackedTensor,
    attention_mask: torch.Tensor,
    num_actions: int,
) -> SamplerHead:
    """Place the packed sampler head rows at the response positions of a left-padded batch.

    Support row ``j`` of a trajectory belongs to its ``j``-th response token, which is predicted by the
    position before it, so the rows are first mapped to logit positions and then shifted by one.
    """
    if sample_support.values.shape[0] != sample_support_logprobs.values.shape[0]:
        raise ValueError(
            f"sample support has {sample_support.values.shape[0]} rows but its logprobs have "
            f"{sample_support_logprobs.values.shape[0]}"
        )
    layout = canonical_token_metadata_layout(attention_mask)
    logit_row_ids = sample_support_row_ids_in_batch_positions(sample_support, layout)
    seq_len = attention_mask.shape[1]
    action_row_ids = logit_row_ids[:, seq_len - num_actions - 1 : seq_len - 1]
    ids = gather_packed_rows(sample_support, action_row_ids, SAMPLE_SUPPORT_PADDING).long()
    logprobs = gather_packed_rows(sample_support_logprobs, action_row_ids, SAMPLE_SUPPORT_LOGPROBS_PADDING).float()
    return SamplerHead(ids=ids, logprobs=logprobs)


def sampled_in_head_fraction(
    head_ids: torch.Tensor, sequences: torch.Tensor, num_actions: int, loss_mask: Optional[torch.Tensor]
) -> float:
    """Fraction of loss-active response tokens whose sampled id is among the recorded head members.

    A head that is aligned to the wrong positions puts this near zero; a correctly aligned head of an
    untruncated sampler sits near one (a sampled token lands outside its top-k head only rarely).
    """
    sampled = sequences[:, -num_actions:]
    in_head = (head_ids == sampled.unsqueeze(-1)).any(dim=-1).float()
    return masked_mean(in_head, loss_mask).item()


def _all_reduce_sum_keep_local_grad(values: torch.Tensor, tp_group) -> torch.Tensor:
    """Sum ``values`` across tensor-parallel ranks while gradients flow only into the local addend."""
    if tp_group is None or torch.distributed.get_world_size(tp_group) == 1:
        return values
    total = values.detach().clone()
    torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    return total + values - values.detach()


def _shard_logits(
    flat_source: torch.Tensor,
    ids: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    lm_head_weight: Optional[torch.Tensor],
    temperature: float,
    chunk_size: Optional[int],
) -> torch.Tensor:
    """Full-vocabulary logits of ``ids`` (``[rows, width]``) from this rank's shard, summed across TP.

    Every valid id lives on exactly one shard, so a masked local gather (or LM-head pair projection on
    the fused path) followed by a TP sum gives the unsharded logit. Negative ids are padding and come back
    as zeros; callers mask them.
    """
    valid = ids >= 0
    local = valid & (ids >= vocab_start_index) & (ids < vocab_end_index)
    local_ids = (ids - vocab_start_index).clamp(0, vocab_end_index - vocab_start_index - 1)
    if lm_head_weight is None:
        values = flat_source.gather(1, local_ids).float()
        values = torch.where(local, values, 0.0)
    else:
        values = _selected_hidden_projection(
            flat_source, local_ids, local, lm_head_weight, temperature, chunk_size, 0.0
        )
    return values


def score_head_members(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    head_ids: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group,
    sampled_logprobs: Optional[torch.Tensor],
    renormalize_over_head: bool = False,
    lm_head_weight: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
    chunk_size: Optional[int] = None,
) -> torch.Tensor:
    """Trainer logprobs of the sampler head members, ``[..., k]``, ``-inf`` on padding members.

    Without ``renormalize_over_head`` the members are scored under the full-vocabulary softmax:
    ``log p_v = l_v - log Z`` with ``log Z = l_y - log p_y`` recovered from the sampled token's logit
    ``l_y`` and its already-computed logprob ``sampled_logprobs`` (same positions as ``sampled_ids``). With
    ``renormalize_over_head`` (sample-support replay, where the trainer's own distribution is the support
    renormalization) ``log Z`` is the log-sum-exp over the head members instead.

    ``logits_or_hidden`` is either this rank's vocabulary shard of the (temperature-scaled) logits, or the
    decoder hidden states when ``lm_head_weight`` is given (the fused LM-head path, where ``temperature``
    is applied inside the projection).
    """
    if logits_or_hidden.shape[:-1] != sampled_ids.shape or head_ids.shape[:-1] != sampled_ids.shape:
        raise ValueError(
            "logits, sampled_ids and head_ids must share their prefix shape, got "
            f"{logits_or_hidden.shape[:-1]}, {sampled_ids.shape} and {head_ids.shape[:-1]}"
        )
    if not renormalize_over_head and sampled_logprobs is None:
        raise ValueError("full-vocabulary head scoring needs the sampled token logprobs")
    if lm_head_weight is not None and lm_head_weight.shape[0] != vocab_end_index - vocab_start_index:
        raise ValueError(
            f"lm_head_weight holds {lm_head_weight.shape[0]} rows for vocabulary shard "
            f"[{vocab_start_index}, {vocab_end_index})"
        )

    flat_source = logits_or_hidden.reshape(-1, logits_or_hidden.shape[-1])
    flat_head = head_ids.reshape(-1, head_ids.shape[-1]).long()
    valid = flat_head >= 0
    kwargs = dict(
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        lm_head_weight=lm_head_weight,
        temperature=temperature,
        chunk_size=chunk_size,
    )
    member_logits = _all_reduce_sum_keep_local_grad(_shard_logits(flat_source, flat_head, **kwargs), tp_group)
    if renormalize_over_head:
        log_z = torch.logsumexp(torch.where(valid, member_logits, float("-inf")), dim=-1)
    else:
        flat_sampled = sampled_ids.reshape(-1, 1).long()
        sampled_logit = _all_reduce_sum_keep_local_grad(_shard_logits(flat_source, flat_sampled, **kwargs), tp_group)
        log_z = sampled_logit.squeeze(1) - sampled_logprobs.reshape(-1).float()
    member_logprobs = torch.where(valid, member_logits - log_z.unsqueeze(-1), float("-inf"))
    return member_logprobs.reshape(head_ids.shape)


def scatter_packed_rows_to_batch(
    model_values: torch.Tensor, layout: TokenMetadataLayout, padding_value: float
) -> torch.Tensor:
    """Scatter packed ``[1, tokens, *row]`` model outputs into canonical ``[batch, seq_len - 1, *row]``.

    Row-shaped counterpart of ``scatter_packed_token_values_to_batch`` for the head members.
    """
    if layout.padded_sequence_lengths is None or layout.cu_seqlens_padded is None:
        raise ValueError("Scattering packed rows requires a packed metadata layout")
    if layout.context_parallel_size > 1:
        raise NotImplementedError("score centering does not support context parallelism yet")
    if model_values.ndim < 2 or model_values.shape[0] != 1:
        raise ValueError(f"Expected packed model values with shape [1, tokens, ...], got {model_values.shape}")
    values = model_values.squeeze(0)
    # Which padded sequence each packed token belongs to, and its offset within it (the THD layout is
    # [seq0, pad0, seq1, pad1, ...]).
    cu_seqlens_padded = layout.cu_seqlens_padded.to(device=values.device, dtype=torch.long)
    token_indices = torch.arange(values.shape[0], device=values.device)
    sequence_indices = torch.searchsorted(cu_seqlens_padded[1:], token_indices, right=True)
    sequence_offsets = token_indices - cu_seqlens_padded[sequence_indices]
    valid_counts = torch.tensor(layout.sequence_lengths, dtype=torch.long, device=values.device) - 1
    packed_mask = sequence_offsets < valid_counts[sequence_indices]

    attention_mask = layout.attention_mask
    token_ordinals = attention_mask.to(torch.long).cumsum(dim=1)
    output_mask = attention_mask[:, :-1] & (
        token_ordinals[:, :-1] < torch.tensor(layout.sequence_lengths, device=values.device).unsqueeze(1)
    )
    batch_values = torch.full(
        (attention_mask.shape[0], attention_mask.shape[1] - 1, *values.shape[1:]),
        padding_value,
        dtype=values.dtype,
        device=values.device,
    )
    batch_values[output_mask] = values[packed_mask]
    return batch_values


def _shard_log_softmax(logits: torch.Tensor, tp_group) -> torch.Tensor:
    """Log-softmax over the full vocabulary from this rank's shard (plain log-softmax without TP)."""
    if tp_group is None or torch.distributed.get_world_size(tp_group) == 1:
        return torch.log_softmax(logits, dim=-1)
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        _compute_distributed_log_softmax,
    )

    return _compute_distributed_log_softmax(logits, group=tp_group)


class _FusedLMHeadLabelAndHeadLogprobs(torch.autograd.Function):
    """Fused LM-head token logprobs plus the logprobs of ``k`` extra ids per position.

    Same chunked structure and numerics as ``FusedLinearChunkedDistributedLogprob`` (per-chunk
    ``logits = hidden @ weight.T`` in the weight dtype, distributed log-softmax in fp32, no full logits
    tensor kept), extended to gather the head members from the *same* per-chunk log-softmax the label
    comes from. That consistency matters: the centering tail mass ``1 - sum_H p_v`` is a few percent,
    so members scored with even bf16-rounding-level differences from the label's normalizer are useless.

    hidden: [B, S, H]; weight: [V//TP, H] (temperature already folded in); target: [B, S] (rolled by
    the caller); head_ids: [B, S, k] with negative entries for padding members (returned as 0, callers
    mask them).
    """

    @staticmethod
    def forward(ctx, hidden, weight, target, head_ids, vocab_start_index, vocab_end_index, chunk_size, tp_group):

        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        masked_target = (target - vocab_start_index).masked_fill(target_mask, 0)
        head_valid = head_ids >= 0
        head_local = head_valid & (head_ids >= vocab_start_index) & (head_ids < vocab_end_index)
        masked_head = (head_ids - vocab_start_index).masked_fill(~head_local, 0)

        seq_size = int(hidden.shape[1])
        num_chunks = (seq_size + chunk_size - 1) // chunk_size
        label_chunks, head_chunks = [], []
        for chunk_idx in range(num_chunks):
            start, end = chunk_idx * chunk_size, min(seq_size, (chunk_idx + 1) * chunk_size)
            logits = torch.matmul(hidden[:, start:end, :].to(weight.dtype), weight.t()).to(dtype=torch.float32)
            log_probs = _shard_log_softmax(logits, tp_group)
            label = torch.gather(log_probs, -1, masked_target[:, start:end].unsqueeze(-1)).squeeze(-1)
            label = label.masked_fill(target_mask[:, start:end], 0.0)
            head = torch.gather(log_probs, -1, masked_head[:, start:end]).masked_fill(~head_local[:, start:end], 0.0)
            label_chunks.append(label)
            head_chunks.append(head)

        # Each valid id lives on exactly one shard, so one SUM per output combines the shards.
        label_log_probs = torch.cat(label_chunks, dim=1)
        head_log_probs = torch.cat(head_chunks, dim=1)
        if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
            torch.distributed.all_reduce(label_log_probs, op=torch.distributed.ReduceOp.SUM, group=tp_group)
            torch.distributed.all_reduce(head_log_probs, op=torch.distributed.ReduceOp.SUM, group=tp_group)

        ctx.save_for_backward(hidden, weight, target_mask, masked_target, head_valid, head_local, masked_head)
        ctx.chunk_size = chunk_size
        ctx.tp_group = tp_group
        return label_log_probs, head_log_probs

    @staticmethod
    def backward(ctx, grad_label, grad_head):

        hidden, weight, target_mask, masked_target, head_valid, head_local, masked_head = ctx.saved_tensors
        chunk_size, tp_group = ctx.chunk_size, ctx.tp_group
        partition_vocab_size, hidden_size = int(weight.shape[0]), int(weight.shape[1])
        seq_size = int(hidden.shape[1])
        num_chunks = (seq_size + chunk_size - 1) // chunk_size

        # d log p_v / d logit_j = delta_{jv} - softmax_j: the softmax term applies on every rank for every
        # scored id (the outputs are TP-combined, so their grads are replicated), the delta only where
        # this rank owns the id. Grads on padding members are dropped.
        grad_head = grad_head.masked_fill(~head_valid, 0.0)
        total_grad = grad_label + grad_head.sum(dim=-1)

        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros((partition_vocab_size, hidden_size), dtype=torch.float32, device=weight.device)
        for chunk_idx in range(num_chunks):
            start, end = chunk_idx * chunk_size, min(seq_size, (chunk_idx + 1) * chunk_size)
            h_chunk = hidden[:, start:end, :]
            logits = torch.matmul(h_chunk.to(weight.dtype), weight.t()).to(dtype=torch.float32)
            grad_logits = _shard_log_softmax(logits, tp_group).exp_()
            grad_logits.neg_().mul_(total_grad[:, start:end].unsqueeze(-1))
            label_grad = grad_label[:, start:end].masked_fill(target_mask[:, start:end], 0.0)
            grad_logits.scatter_add_(-1, masked_target[:, start:end].unsqueeze(-1), label_grad.unsqueeze(-1))
            head_grad = grad_head[:, start:end].masked_fill(~head_local[:, start:end], 0.0)
            grad_logits.scatter_add_(-1, masked_head[:, start:end], head_grad)

            grad_logits = grad_logits.to(dtype=weight.dtype)
            grad_hidden[:, start:end, :] = torch.matmul(grad_logits, weight)
            grad_weight.add_(
                torch.matmul(
                    grad_logits.reshape(-1, partition_vocab_size).t(),
                    h_chunk.reshape(-1, hidden_size).to(dtype=grad_logits.dtype),
                ).to(torch.float32)
            )
        return grad_hidden, grad_weight.to(weight.dtype), None, None, None, None, None, None


def fused_label_and_head_logprobs(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    target: torch.Tensor,
    head_ids: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group,
    temperature: float = 1.0,
    chunk_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Label logprobs ``[B, S]`` and head-member logprobs ``[B, S, k]`` (``-inf`` on padding members)
    from decoder hidden states and the LM-head weight, in one chunked pass with consistent numerics."""
    if temperature != 1.0:
        lm_head_weight = lm_head_weight / temperature
    seq_size = int(hidden.shape[1])
    effective_chunk = chunk_size if (chunk_size is not None and 0 < chunk_size < seq_size) else max(seq_size, 1)
    label_log_probs, head_log_probs = _FusedLMHeadLabelAndHeadLogprobs.apply(
        hidden,
        lm_head_weight,
        target.long(),
        head_ids.long(),
        vocab_start_index,
        vocab_end_index,
        effective_chunk,
        tp_group,
    )
    return label_log_probs, torch.where(head_ids >= 0, head_log_probs, float("-inf"))


def compute_score_centering_head_logprobs(
    logits_or_hidden: torch.Tensor,
    sequences: torch.Tensor,
    sample_support: PackedTensor,
    token_logprobs: Optional[torch.Tensor],
    num_actions: int,
    *,
    packed: bool,
    metadata_layout: TokenMetadataLayout,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group,
    lm_head_weight: Optional[torch.Tensor],
    temperature: float,
    chunk_size: Optional[int],
    renormalize_over_head: bool = False,
) -> tuple[Optional[torch.Tensor], torch.Tensor]:
    """Megatron entry point: trainer logprobs of the sampler head at every response position.

    Aligns the packed support rows to the model's token layout (padded or THD-packed) and returns
    ``(token_logprobs, head_logprobs)`` in canonical batch order, ``head_logprobs`` as
    ``[batch, num_actions, k]`` with ``-inf`` on padding members.

    On the fused LM-head path (``lm_head_weight`` given) the label and head logprobs come out of one
    chunked pass (``fused_label_and_head_logprobs``), and the returned ``token_logprobs``
    (``[batch, seq_len - 1]``) replace the caller's standard label logprobs so both share one normalizer;
    ``token_logprobs`` may then be ``None`` on input. With materialized logits the members are gathered
    from ``logits_or_hidden`` and normalized with the caller's ``token_logprobs``
    (``log Z = l_y - log p_y``), and ``None`` is returned in the first slot.
    """
    if packed:
        row_ids = align_sample_support_row_ids(sample_support, metadata_layout)
        aligned_sampled_ids = align_token_metadata(sequences, metadata_layout, 0, next_token=True)
        aligned_source = logits_or_hidden
    else:
        row_ids = sample_support_row_ids_in_batch_positions(sample_support, metadata_layout)[:, :-1]
        aligned_sampled_ids = sequences[:, 1:]
        aligned_source = logits_or_hidden[:, :-1]
    head_ids = gather_packed_rows(sample_support, row_ids, SAMPLE_SUPPORT_PADDING).long()
    valid = head_ids >= 0

    if lm_head_weight is not None:
        if renormalize_over_head:
            raise NotImplementedError(
                "score centering with sample-support replay is not supported on the fused LM-head path"
            )
        label_log_probs, member_logprobs = fused_label_and_head_logprobs(
            aligned_source,
            lm_head_weight,
            aligned_sampled_ids,
            head_ids,
            vocab_start_index=vocab_start_index,
            vocab_end_index=vocab_end_index,
            tp_group=tp_group,
            temperature=temperature,
            chunk_size=chunk_size,
        )
        if packed:
            label_log_probs = scatter_packed_rows_to_batch(label_log_probs.unsqueeze(-1), metadata_layout, 0.0).squeeze(
                -1
            )
            member_logprobs = scatter_packed_rows_to_batch(member_logprobs, metadata_layout, float("-inf"))
        return label_log_probs, member_logprobs[:, -num_actions:]

    flat_source = aligned_source.reshape(-1, aligned_source.shape[-1])
    kwargs = dict(
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        lm_head_weight=None,
        temperature=temperature,
        chunk_size=chunk_size,
    )
    flat_head = head_ids.reshape(-1, head_ids.shape[-1])
    member_logits = _all_reduce_sum_keep_local_grad(_shard_logits(flat_source, flat_head, **kwargs), tp_group)
    member_logits = member_logits.reshape(head_ids.shape)
    if renormalize_over_head:
        log_z = torch.logsumexp(torch.where(valid, member_logits, float("-inf")), dim=-1)
        member_logprobs = torch.where(valid, member_logits - log_z.unsqueeze(-1), float("-inf"))
        if packed:
            member_logprobs = scatter_packed_rows_to_batch(member_logprobs, metadata_layout, float("-inf"))
        return None, member_logprobs[:, -num_actions:]

    if token_logprobs is None:
        raise ValueError("materialized-logits head scoring needs the caller's sampled-token logprobs")
    sampled_logit = _all_reduce_sum_keep_local_grad(
        _shard_logits(flat_source, aligned_sampled_ids.reshape(-1, 1).long(), **kwargs), tp_group
    ).reshape(aligned_sampled_ids.shape)
    if packed:
        # Combine in canonical batch order, where the sampled-token logprobs already live.
        member_logits = scatter_packed_rows_to_batch(member_logits, metadata_layout, 0.0)
        valid = scatter_packed_rows_to_batch(valid, metadata_layout, False)
        sampled_logit = scatter_packed_rows_to_batch(sampled_logit.unsqueeze(-1), metadata_layout, 0.0).squeeze(-1)
    member_logits = member_logits[:, -num_actions:]
    valid = valid[:, -num_actions:]
    sampled_logit = sampled_logit[:, -num_actions:]
    log_z = sampled_logit - token_logprobs[:, -num_actions:].float()
    return None, torch.where(valid, member_logits - log_z.unsqueeze(-1), float("-inf"))
