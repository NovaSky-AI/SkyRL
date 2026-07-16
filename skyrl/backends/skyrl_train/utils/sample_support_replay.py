"""Support-conditioned logprobs for bounded sampler replay."""

import torch

from skyrl.utils.token_metadata import (
    TokenMetadataLayout,
    align_token_metadata,
    scatter_packed_token_values_to_batch,
)


def _selected_hidden_projection(
    hidden: torch.Tensor,
    token_ids: torch.Tensor,
    local_mask: torch.Tensor,
    lm_head_weight: torch.Tensor,
    temperature: float,
    chunk_size: int | None,
    invalid_value: float,
) -> torch.Tensor:
    """Project selected candidate pairs without materializing vocabulary logits."""
    num_rows, width = token_ids.shape
    row_ids = torch.arange(num_rows, device=hidden.device).unsqueeze(1).expand(-1, width).reshape(-1)
    flat_token_ids = token_ids.reshape(-1)
    flat_mask = local_mask.reshape(-1)
    output = torch.empty(flat_token_ids.shape, dtype=torch.float32, device=hidden.device)
    # Bound the temporary [candidate pairs, hidden] projection for wide supports.
    pair_chunk_size = flat_token_ids.numel() if chunk_size is None else chunk_size
    for start in range(0, flat_token_ids.numel(), pair_chunk_size):
        end = min(start + pair_chunk_size, flat_token_ids.numel())
        selected_hidden = hidden.index_select(0, row_ids[start:end]).to(lm_head_weight.dtype)
        selected_weight = lm_head_weight.index_select(0, flat_token_ids[start:end])
        projected = (selected_hidden * selected_weight).sum(dim=-1) / temperature
        output[start:end] = torch.where(
            flat_mask[start:end],
            projected.to(torch.float32),
            invalid_value,
        )
    return output.reshape(num_rows, width)


def sample_support_logprobs(
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Renormalize sampled-token scores over each recorded support row."""
    if logits_or_hidden.shape[:-1] != sampled_ids.shape or support_ids.shape[:-1] != sampled_ids.shape:
        raise ValueError(
            "logits, sampled_ids, and support_ids must have matching prefix shapes, got "
            f"{logits_or_hidden.shape[:-1]}, {sampled_ids.shape}, and {support_ids.shape[:-1]}"
        )
    if support_ids.dtype != torch.int32:
        raise ValueError(f"sample support must use int32 vocab ids, got {support_ids.dtype}")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

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
            raise ValueError("lm_head_weight rows do not match the configured vocabulary shard")
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
    # Numerator and denominator share one TP SUM collective.
    local_stats = torch.stack((local_sum, local_sampled))
    global_stats = local_stats.detach().clone()
    if tp_group is not None and torch.distributed.get_world_size(tp_group) > 1:
        torch.distributed.all_reduce(global_stats, op=torch.distributed.ReduceOp.SUM, group=tp_group)
    global_stats = global_stats + local_stats - local_stats.detach()
    denominator, sampled_score = global_stats
    logprobs = sampled_score - safe_max - torch.where(valid_rows, denominator, 1.0).log()
    logprobs = torch.where(valid_rows, logprobs, 0.0)
    return logprobs.reshape(sampled_ids.shape), valid_rows.reshape(sampled_ids.shape)


def synthetic_eos_logprobs(
    logits_or_hidden: torch.Tensor,
    sampled_ids: torch.Tensor,
    synthetic_eos_mask: torch.Tensor,
    *,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup,
    inference_only: bool,
    lm_head_weight: torch.Tensor | None = None,
    temperature: float = 1.0,
    chunk_size: int | None = None,
    fused_backend: str = "torch",
    metadata_layout: TokenMetadataLayout | None = None,
) -> torch.Tensor:
    """Compute ordinary logprobs for EOS tokens appended after vLLM generation."""
    if synthetic_eos_mask.shape != sampled_ids.shape:
        raise ValueError("synthetic_eos_mask and sampled_ids must have matching shapes")

    if metadata_layout is not None and metadata_layout.padded_sequence_lengths is not None:
        if synthetic_eos_mask.shape[0] != 1:
            raise ValueError("Packed synthetic EOS metadata must have a singleton batch dimension")
        if metadata_layout.cu_seqlens_padded is None:
            raise ValueError("Packed synthetic EOS fallback requires padded sequence boundaries")
        if any(length <= 0 for length in metadata_layout.padded_sequence_lengths):
            raise ValueError("Synthetic EOS fallback requires non-empty trajectory segments")
        expected_tokens = metadata_layout.aligned_sequence_length // metadata_layout.context_parallel_size
        if expected_tokens != synthetic_eos_mask.numel():
            raise ValueError("Synthetic EOS layout does not match the model token layout")
        lengths = (
            metadata_layout.cu_seqlens_padded.to(
                device=synthetic_eos_mask.device,
                dtype=torch.long,
            ).diff()
            // metadata_layout.context_parallel_size
        )
    else:
        if synthetic_eos_mask.shape[0] == 0 or synthetic_eos_mask.shape[1] == 0:
            raise ValueError("Synthetic EOS fallback requires non-empty trajectory segments")
        lengths = torch.full(
            (synthetic_eos_mask.shape[0],),
            synthetic_eos_mask.shape[1],
            dtype=torch.long,
            device=synthetic_eos_mask.device,
        )

    # Preprocessing permits at most one unsupported loss-bearing EOS per
    # trajectory. Select one fixed slot for every trajectory so TP collectives
    # never depend on the number of EOS fallbacks in this microbatch.
    offsets = lengths.cumsum(dim=0) - lengths
    trajectory_ids = torch.repeat_interleave(
        torch.arange(lengths.shape[0], device=lengths.device),
        lengths,
        output_size=synthetic_eos_mask.numel(),
    )
    token_indices = torch.arange(synthetic_eos_mask.numel(), device=lengths.device)
    sentinel = synthetic_eos_mask.numel()
    candidate_indices = torch.where(synthetic_eos_mask.reshape(-1), token_indices, sentinel)
    selected_indices = torch.full_like(offsets, sentinel).scatter_reduce(
        0,
        trajectory_ids,
        candidate_indices,
        reduce="amin",
        include_self=True,
    )
    has_selection = selected_indices != sentinel
    selected_indices = torch.where(has_selection, selected_indices, offsets)

    flat_source = logits_or_hidden.reshape(-1, logits_or_hidden.shape[-1])
    flat_targets = sampled_ids.reshape(-1)
    selected_source = flat_source.index_select(0, selected_indices)
    selected_targets = flat_targets.index_select(0, selected_indices)
    if lm_head_weight is None:
        from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
            DistributedLogprob,
        )

        selected = DistributedLogprob.apply(
            selected_source.unsqueeze(0),
            selected_targets.unsqueeze(0),
            vocab_start_index,
            vocab_end_index,
            tp_group,
            inference_only,
        ).squeeze(0)
    else:
        from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
            _fused_lm_head_logprob_apply,
        )

        if temperature != 1.0:
            lm_head_weight = lm_head_weight / temperature
        selected_chunk_size = (
            selected_source.shape[0] if chunk_size is None else min(chunk_size, selected_source.shape[0])
        )
        selected = _fused_lm_head_logprob_apply(
            fused_backend,
            selected_source.unsqueeze(0),
            lm_head_weight,
            selected_targets.unsqueeze(0),
            vocab_start_index,
            vocab_end_index,
            selected_chunk_size,
            tp_group,
            inference_only,
        ).squeeze(0)
    selected = torch.where(has_selection, selected, 0.0).to(torch.float32)
    output = torch.zeros(sampled_ids.numel(), dtype=torch.float32, device=logits_or_hidden.device)
    output = output.scatter_add(0, selected_indices, selected)
    return output.reshape(sampled_ids.shape)


def compute_sample_support_logprobs(
    logits_or_hidden: torch.Tensor,
    sequences: torch.Tensor,
    loss_mask: torch.Tensor | None,
    sample_support_ids: torch.Tensor | None,
    num_actions: int,
    *,
    packed: bool,
    metadata_layout: TokenMetadataLayout | None,
    vocab_start_index: int,
    vocab_end_index: int,
    tp_group: torch.distributed.ProcessGroup,
    inference_only: bool,
    lm_head_weight: torch.Tensor | None,
    temperature: float,
    chunk_size: int | None,
    fused_backend: str,
) -> torch.Tensor:
    """Compute support-conditioned logprobs in canonical trainer layout."""
    if sample_support_ids is None:
        raise ValueError("sample-support replay is enabled but the microbatch has no recorded support")
    if loss_mask is None:
        raise ValueError("sample-support replay requires the response loss mask")

    target_loss_mask = torch.zeros_like(sequences, dtype=torch.bool)
    target_loss_mask[:, -num_actions:] = loss_mask.to(torch.bool)
    if packed:
        if metadata_layout is None:
            raise ValueError("Packed sample-support replay requires the shared token metadata layout")
        aligned_sampled_ids = align_token_metadata(sequences, metadata_layout, 0, next_token=True)
        aligned_support_ids = align_token_metadata(sample_support_ids, metadata_layout, -1, next_token=True)
        aligned_loss_mask = align_token_metadata(target_loss_mask, metadata_layout, False, next_token=True)
    else:
        aligned_sampled_ids = sequences[:, 1:]
        aligned_support_ids = sample_support_ids[:, 1:]
        aligned_loss_mask = target_loss_mask[:, 1:]

    support_logprobs, valid_support = sample_support_logprobs(
        logits_or_hidden if packed else logits_or_hidden[:, :-1],
        aligned_sampled_ids,
        aligned_support_ids,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        tp_group=tp_group,
        lm_head_weight=lm_head_weight,
        temperature=temperature if lm_head_weight is not None else 1.0,
        chunk_size=chunk_size,
    )
    # Preprocessing permits an empty loss-bearing row only for an EOS that SkyRL
    # appended after generation. vLLM never supplied a support set for that token.
    synthetic_eos_mask = aligned_loss_mask & ~valid_support
    eos_logprobs = synthetic_eos_logprobs(
        logits_or_hidden if packed else logits_or_hidden[:, :-1],
        aligned_sampled_ids,
        synthetic_eos_mask,
        vocab_start_index=vocab_start_index,
        vocab_end_index=vocab_end_index,
        tp_group=tp_group,
        inference_only=inference_only,
        lm_head_weight=lm_head_weight,
        temperature=temperature if lm_head_weight is not None else 1.0,
        chunk_size=chunk_size,
        fused_backend=fused_backend,
        metadata_layout=metadata_layout if packed else None,
    )
    token_logprobs = torch.where(synthetic_eos_mask, eos_logprobs, support_logprobs)
    return scatter_packed_token_values_to_batch(token_logprobs, metadata_layout, 0) if packed else token_logprobs
