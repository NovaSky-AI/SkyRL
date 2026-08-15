"""Support-conditioned scoring for bounded sampler replay."""

from dataclasses import dataclass

import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
    align_token_metadata,
    scatter_packed_token_values_to_batch,
)
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    SAMPLE_SUPPORT_NO_ROW,
    SAMPLE_SUPPORT_PADDING,
    SAMPLE_SUPPORT_TORCH_DTYPE,
    align_sample_support_row_ids,
)


def missing_sample_support_message(backend: str) -> str:
    """Describe how to enable and forward a missing support payload."""
    return (
        f"sample-support replay is enabled but the {backend} forward received no "
        f"{SAMPLE_SUPPORT_FIELD!r}. Set generator.inference_engine.enable_return_sample_support_set=true "
        "so the sampler records it; if it is already set, the generator is dropping the field before "
        "the trainer sees it -- SkyRLGymGenerator forwards it, and a custom generator must too."
    )


@dataclass(frozen=True)
class SampleSupportScores:
    """Support-conditioned scores and the rows a recorded support row backs."""

    logprobs: torch.Tensor
    entropy: torch.Tensor | None
    valid_mask: torch.Tensor


def reject_unsupported_sample_support_packing(sub_seq_lengths: list[list[int]] | None) -> None:
    """Require every packed row to contain one whole trajectory."""
    if sub_seq_lengths is None:
        return
    if any(len(row_lengths) > 1 for row_lengths in sub_seq_lengths):
        raise ValueError(
            "sample-support replay does not support controller-packed multi-subsequence rows, got "
            f"sub-sequence counts {[len(row_lengths) for row_lengths in sub_seq_lengths]}"
        )


def _project_pair_chunk(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    row_ids: torch.Tensor,
    token_ids: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Score one bounded block of ``(token position, vocab row)`` candidate pairs."""
    selected_hidden = hidden.index_select(0, row_ids).to(weight.dtype)
    selected_weight = weight.index_select(0, token_ids)
    return (selected_hidden * selected_weight).sum(dim=-1) / temperature


class _ChunkedCandidateProjection(torch.autograd.Function):
    """Selected LM-head projection that never retains a ``[pairs, hidden]`` activation."""

    @staticmethod
    def forward(ctx, hidden, weight, row_ids, token_ids, temperature, chunk_size):
        ctx.save_for_backward(hidden, weight, row_ids, token_ids)
        ctx.temperature = temperature
        ctx.chunk_size = chunk_size
        output = torch.empty(token_ids.shape, dtype=torch.float32, device=hidden.device)
        for start in range(0, token_ids.numel(), chunk_size):
            end = min(start + chunk_size, token_ids.numel())
            projected = _project_pair_chunk(hidden, weight, row_ids[start:end], token_ids[start:end], temperature)
            output[start:end] = projected.to(torch.float32)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, row_ids, token_ids = ctx.saved_tensors
        grad_hidden = torch.zeros_like(hidden) if ctx.needs_input_grad[0] else None
        grad_weight = torch.zeros_like(weight) if ctx.needs_input_grad[1] else None
        for start in range(0, token_ids.numel(), ctx.chunk_size):
            end = min(start + ctx.chunk_size, token_ids.numel())
            chunk_rows = row_ids[start:end]
            chunk_tokens = token_ids[start:end]
            chunk_grad = grad_output[start:end].to(weight.dtype) / ctx.temperature
            if grad_hidden is not None:
                hidden_contribution = chunk_grad.unsqueeze(1) * weight.index_select(0, chunk_tokens)
                grad_hidden.index_add_(0, chunk_rows, hidden_contribution.to(hidden.dtype))
            if grad_weight is not None:
                selected_hidden = hidden.index_select(0, chunk_rows).to(weight.dtype)
                grad_weight.index_add_(0, chunk_tokens, chunk_grad.unsqueeze(1) * selected_hidden)
        return grad_hidden, grad_weight, None, None, None, None


def _project_candidate_pairs(
    hidden: torch.Tensor,
    row_ids: torch.Tensor,
    token_ids: torch.Tensor,
    lm_head_weight: torch.Tensor,
    temperature: float,
    chunk_size: int | None,
) -> torch.Tensor:
    """Project selected ``(token position, vocab row)`` pairs in bounded chunks."""
    if token_ids.numel() == 0:
        return torch.empty(0, dtype=torch.float32, device=hidden.device)
    pair_chunk_size = token_ids.numel() if chunk_size is None else chunk_size
    if pair_chunk_size <= 0:
        raise ValueError(f"candidate projection chunk size must be positive, got {pair_chunk_size}")
    if torch.is_grad_enabled() and (hidden.requires_grad or lm_head_weight.requires_grad):
        # A plain loop retains every selected chunk for backward. The custom backward reselects
        # one chunk at a time, so peak candidate-pair storage stays at the chunk bound.
        return _ChunkedCandidateProjection.apply(
            hidden,
            lm_head_weight,
            row_ids,
            token_ids,
            temperature,
            pair_chunk_size,
        )

    output = torch.empty(token_ids.shape, dtype=torch.float32, device=hidden.device)
    for start in range(0, token_ids.numel(), pair_chunk_size):
        end = min(start + pair_chunk_size, token_ids.numel())
        projected = _project_pair_chunk(hidden, lm_head_weight, row_ids[start:end], token_ids[start:end], temperature)
        output[start:end] = projected.to(torch.float32)
    return output


def _selected_hidden_projection(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    local_mask: torch.Tensor,
    lm_head_weight: torch.Tensor,
    temperature: float,
    chunk_size: int | None,
    invalid_value: float,
) -> torch.Tensor:
    """Score a fixed-width candidate matrix without materializing vocabulary logits."""
    num_rows, width = token_ids.shape
    row_ids = torch.arange(num_rows, device=hidden.device).unsqueeze(1).expand(-1, width).reshape(-1)
    projected = _project_candidate_pairs(
        hidden,
        row_ids,
        token_ids.reshape(-1),
        lm_head_weight,
        temperature,
        chunk_size,
    )
    return torch.where(local_mask.reshape(-1), projected, invalid_value).reshape(num_rows, width)


def sample_support_scores(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    support_ids: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
) -> SampleSupportScores:
    """Renormalize each sampled token's score over its recorded support row."""
    if logits_or_hidden.shape[:-1] != sampled_ids.shape or support_ids.shape[:-1] != sampled_ids.shape:
        raise ValueError(
            "logits, sampled_ids, and support_ids must have matching prefix shapes, got "
            f"{logits_or_hidden.shape[:-1]}, {sampled_ids.shape}, and {support_ids.shape[:-1]}"
        )
    if support_ids.dtype != SAMPLE_SUPPORT_TORCH_DTYPE:
        raise ValueError(f"sample support must use {SAMPLE_SUPPORT_TORCH_DTYPE} vocab ids, got {support_ids.dtype}")
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")

    flat_source = logits_or_hidden.reshape(-1, logits_or_hidden.shape[-1])
    flat_sampled = sampled_ids.reshape(-1).long()
    flat_support = support_ids.reshape(-1, support_ids.shape[-1]).long()
    valid_members = flat_support >= 0
    valid_rows = valid_members.any(dim=-1)
    local_members = valid_members & (flat_support >= vocab_start_index) & (flat_support < vocab_end_index)
    local_support_ids = (flat_support - vocab_start_index).clamp(0, vocab_end_index - vocab_start_index - 1)
    local_sample_mask = (flat_sampled >= vocab_start_index) & (flat_sampled < vocab_end_index)
    local_sample_ids = (flat_sampled - vocab_start_index).clamp(0, vocab_end_index - vocab_start_index - 1)

    compute_dtype = (
        torch.float32 if logits_or_hidden.dtype in (torch.float16, torch.bfloat16) else logits_or_hidden.dtype
    )
    if lm_head_weight is None:
        local_values = flat_source.gather(1, local_support_ids).to(compute_dtype)
        local_values = torch.where(local_members, local_values, float("-inf"))
        local_sampled = flat_source.gather(1, local_sample_ids.unsqueeze(1)).squeeze(1).to(compute_dtype)
        local_sampled = torch.where(local_sample_mask, local_sampled, 0.0)
    else:
        if lm_head_weight.shape[0] != vocab_end_index - vocab_start_index:
            raise ValueError(
                f"lm_head_weight holds {lm_head_weight.shape[0]} rows for vocabulary shard "
                f"[{vocab_start_index}, {vocab_end_index})"
            )
        local_values = _selected_hidden_projection(
            flat_source,
            local_support_ids,
            local_members,
            lm_head_weight,
            temperature,
            chunk_size,
            float("-inf"),
        )
        local_sampled = _selected_hidden_projection(
            flat_source,
            local_sample_ids.unsqueeze(1),
            local_sample_mask.unsqueeze(1),
            lm_head_weight,
            temperature,
            chunk_size,
            0.0,
        ).squeeze(1)

    local_max = local_values.detach().amax(dim=-1)
    global_max = local_max.clone()
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(global_max, op=torch.distributed.ReduceOp.MAX, group=tp_group)
    safe_max = torch.where(valid_rows, global_max, 0.0)

    local_sum = torch.where(local_members, (local_values - safe_max.unsqueeze(1)).exp(), 0.0).sum(dim=-1)
    # Denominator and numerator share one SUM collective, so a TP scorer costs two reductions.
    local_stats = torch.stack((local_sum, local_sampled))
    global_stats = local_stats.detach().clone()
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(global_stats, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    global_stats = global_stats + local_stats - local_stats.detach()
    denominator, sampled_score = global_stats
    logprobs = sampled_score - safe_max - torch.where(valid_rows, denominator, 1.0).log()
    return SampleSupportScores(
        logprobs=torch.where(valid_rows, logprobs, 0.0).reshape(sampled_ids.shape),
        entropy=None,
        valid_mask=valid_rows.reshape(sampled_ids.shape),
    )


def sample_support_row_ids_in_batch_positions(
    sample_support: PackedTensor,
    layout: TokenMetadataLayout,
) -> torch.Tensor:
    """Map support rows into canonical left-padded batch coordinates."""
    row_ids = align_sample_support_row_ids(sample_support, layout)
    sequence_length = layout.attention_mask.shape[1]
    padding_widths = sequence_length - layout.attention_mask.sum(dim=1).to(torch.long)
    positions = torch.arange(sequence_length, device=row_ids.device).unsqueeze(0) - padding_widths.unsqueeze(1)
    return torch.where(positions >= 0, row_ids.gather(1, positions.clamp(min=0)), SAMPLE_SUPPORT_NO_ROW)


def _gather_support_rows(support: PackedTensor, row_ids: torch.Tensor) -> torch.Tensor:
    """Gather each model position's ``[top_k]`` support row, by id rather than by placement."""
    top_k = support.values.shape[1]
    if support.values.shape[0] == 0:
        return torch.full(
            (*row_ids.shape, top_k),
            SAMPLE_SUPPORT_PADDING,
            dtype=support.dtype,
            device=support.device,
        )
    flat_row_ids = row_ids.reshape(-1)
    gathered = support.values.index_select(0, flat_row_ids.clamp(min=0))
    gathered = gathered.masked_fill((flat_row_ids < 0).unsqueeze(1), SAMPLE_SUPPORT_PADDING)
    return gathered.reshape(*row_ids.shape, top_k)


def score_aligned_sample_support(
    aligned_source: torch.Tensor,
    aligned_sampled_ids: torch.Tensor,
    aligned_row_ids: torch.Tensor,
    aligned_loss_mask: torch.Tensor,
    sample_support: PackedTensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
) -> SampleSupportScores:
    """Score token-aligned recorded support rows."""
    scores = sample_support_scores(
        aligned_source,
        aligned_sampled_ids,
        _gather_support_rows(sample_support, aligned_row_ids),
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        tp_group=tp_group,
        lm_head_weight=lm_head_weight,
        temperature=temperature if lm_head_weight is not None else 1.0,
        chunk_size=chunk_size,
    )
    unsupported_loss_active = aligned_loss_mask & ~scores.valid_mask
    if unsupported_loss_active.any():
        raise ValueError("sample-support replay requires captured support for every loss-active token")
    return scores


def compute_sample_support_scores(
    logits_or_hidden: torch.Tensor,
    sequences: torch.Tensor,
    loss_mask: torch.Tensor | None,
    sample_support: PackedTensor | None,
    num_actions: int,
    *,
    packed: bool,
    metadata_layout: TokenMetadataLayout | None,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup | None,
    lm_head_weight: torch.Tensor | None,
    temperature: float,
    chunk_size: int | None,
) -> SampleSupportScores:
    """Score support-conditioned logprobs in canonical trainer layout."""
    if sample_support is None:
        raise ValueError(missing_sample_support_message("Megatron"))
    if loss_mask is None:
        raise ValueError("sample-support replay requires the response loss mask")
    if metadata_layout is None:
        raise ValueError("sample-support replay requires the shared token metadata layout")

    target_loss_mask = torch.zeros_like(sequences, dtype=torch.bool)
    target_loss_mask[:, -num_actions:] = loss_mask.to(torch.bool)
    if packed:
        row_ids = align_sample_support_row_ids(sample_support, metadata_layout)
        aligned_sampled_ids = align_token_metadata(sequences, metadata_layout, 0, next_token=True)
        aligned_loss_mask = align_token_metadata(target_loss_mask, metadata_layout, False, next_token=True)
        aligned_source = logits_or_hidden
    else:
        # The domain ends at real token L_i - 2, so dropping the last column loses no row.
        row_ids = sample_support_row_ids_in_batch_positions(sample_support, metadata_layout)[:, :-1]
        aligned_sampled_ids = sequences[:, 1:]
        aligned_loss_mask = target_loss_mask[:, 1:]
        aligned_source = logits_or_hidden[:, :-1]

    scores = score_aligned_sample_support(
        aligned_source,
        aligned_sampled_ids,
        row_ids,
        aligned_loss_mask,
        sample_support,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        tp_group=tp_group,
        lm_head_weight=lm_head_weight,
        temperature=temperature,
        chunk_size=chunk_size,
    )
    if not packed:
        return scores
    return SampleSupportScores(
        logprobs=scatter_packed_token_values_to_batch(scores.logprobs, metadata_layout, 0),
        entropy=None,
        valid_mask=scatter_packed_token_values_to_batch(scores.valid_mask, metadata_layout, False),
    )
