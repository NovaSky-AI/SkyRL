"""
Utility functions for MoE Router Replay.
"""

from typing import List

import torch


def patch_topk_router_layer_number():
    """Monkey-patch TopKRouter.set_layer_number to propagate the global layer
    number to the RouterReplay instance.

    DeepSeek V3 (and similar) architectures have dense FFN layers before the MoE
    layers.  vLLM reports routing indices for ALL transformer layers (including
    dense), but Megatron only creates RouterReplay instances for MoE layers.
    Storing the global layer_number on each RouterReplay instance lets us map
    vLLM's per-layer data to the correct MoE router even when dense layers are
    present.

    Must be called BEFORE model creation (i.e. before make_megatron_module).
    """
    try:
        from megatron.core.transformer.moe.router import TopKRouter
    except ImportError:
        return

    if getattr(TopKRouter, "_set_layer_number_patched", False):
        return

    original_set_layer_number = TopKRouter.set_layer_number

    def patched_set_layer_number(self, layer_number: int):
        original_set_layer_number(self, layer_number)
        if self.router_replay is not None:
            self.router_replay.layer_number = layer_number

    TopKRouter.set_layer_number = patched_set_layer_number
    TopKRouter._set_layer_number_patched = True


def _patch_alltoall_dispatcher_for_replay():
    """Monkey-patch MoEAlltoAllTokenDispatcher.preprocess to handle router replay.

    When router replay is enabled, duplicate indices in top_indices can cause
    routing_map.sum() < num_tokens * topk, leading to a split size mismatch
    in the alltoall collective.  We fix this by deriving num_out_tokens from
    the routing map instead of the static num_tokens * topk formula.

    Reference: https://github.com/verl-project/verl/pull/4986
    """
    try:
        from megatron.core.transformer.moe.token_dispatcher import (
            MoEAlltoAllTokenDispatcher,
        )
    except ImportError:
        return

    if getattr(MoEAlltoAllTokenDispatcher, "_preprocess_patched", False):
        return

    original_preprocess = MoEAlltoAllTokenDispatcher.preprocess

    def patched_preprocess(self, routing_map):
        result = original_preprocess(self, routing_map)
        if (
            getattr(self.config, "moe_enable_routing_replay", False)
            and not self.drop_and_pad
            and self.config.moe_expert_capacity_factor is None
            and not self.config.moe_router_padding_for_quantization
        ):
            self.num_out_tokens = int(routing_map.sum().item())
        return result

    MoEAlltoAllTokenDispatcher.preprocess = patched_preprocess
    MoEAlltoAllTokenDispatcher._preprocess_patched = True


def _split_replay_indices(rollout_expert_indices: torch.Tensor) -> List[torch.Tensor]:
    if rollout_expert_indices is None:
        return None
    if rollout_expert_indices.dim() != 4:
        raise ValueError(f"Expected 4D replay indices, got shape {rollout_expert_indices.shape}")
    per_layer = rollout_expert_indices.permute(2, 0, 1, 3).contiguous()
    # flatten [batch, seq, topk] to [batch * seq, topk] for each layer
    return [per_layer[i].reshape(-1, per_layer.shape[-1]) for i in range(per_layer.shape[0])]


def _remove_left_padding_from_indices(
    rollout_expert_indices: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the same left-padding removal as remove_left_padding to routing indices.

    Args:
        rollout_expert_indices: [batch, padded_seq_len, layers, topk]
        attention_mask: [batch, padded_seq_len] (int or bool)

    Returns:
        [batch, effective_seq_len, layers, topk] with real tokens packed left.
    """
    import megatron.core.parallel_state as mpu

    seq_lens = attention_mask.sum(dim=1)
    effective_seq_len = seq_lens.max().item()
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
    if align_size > 1:
        pad_size = (align_size - effective_seq_len % align_size) % align_size
        effective_seq_len += pad_size

    batch_size = rollout_expert_indices.shape[0]
    new_rii = torch.zeros(
        batch_size,
        effective_seq_len,
        rollout_expert_indices.shape[2],
        rollout_expert_indices.shape[3],
        dtype=rollout_expert_indices.dtype,
        device=rollout_expert_indices.device,
    )
    for i in range(batch_size):
        mask = attention_mask[i].bool()
        new_rii[i, : seq_lens[i]] = rollout_expert_indices[i, mask]
    return new_rii


def _pack_replay_indices(
    rollout_expert_indices: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pack routing indices to match the token layout produced by preprocess_packed_seqs.

    With sample packing, Megatron concatenates all sequences into one packed
    sequence with per-sample alignment padding.  The MoE router sees tokens in
    this packed order, so replay indices must follow the same layout.

    Returns:
        [1, total_packed_len, layers, topk] matching the packed model input.
    """
    import megatron.core.parallel_state as mpu

    batch_size = rollout_expert_indices.shape[0]
    num_layers = rollout_expert_indices.shape[2]
    topk = rollout_expert_indices.shape[3]

    seq_lens = attention_mask.sum(dim=-1, dtype=torch.int32)
    tp_size = mpu.get_tensor_model_parallel_world_size()
    cp_size = mpu.get_context_parallel_world_size()
    align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size

    pad_sizes = (align_size - seq_lens % align_size) % align_size
    seqlens_padded = seq_lens + pad_sizes

    total_packed_len = int(seqlens_padded.sum().item())

    packed = torch.zeros(
        total_packed_len,
        num_layers,
        topk,
        dtype=rollout_expert_indices.dtype,
        device=rollout_expert_indices.device,
    )

    seq_lens_cpu = seq_lens.tolist()
    seqlens_padded_cpu = seqlens_padded.tolist()
    offset = 0
    for i in range(batch_size):
        n = seq_lens_cpu[i]
        mask = attention_mask[i].bool()
        d = rollout_expert_indices[i, mask]
        packed[offset : offset + n] = d
        offset += seqlens_padded_cpu[i]

    if cp_size > 1:
        cp_rank = mpu.get_context_parallel_rank()
        out = torch.zeros(
            total_packed_len // cp_size,
            num_layers,
            topk,
            dtype=packed.dtype,
            device=packed.device,
        )
        src_offset = 0
        dst_offset = 0
        for i in range(batch_size):
            seqlen_padded_i = seqlens_padded_cpu[i]
            seqlen_per_cp = seqlen_padded_i // cp_size
            half = seqlen_per_cp // 2
            out[dst_offset : dst_offset + half] = packed[
                src_offset + half * cp_rank : src_offset + half * (cp_rank + 1)
            ]
            back_start = src_offset + seqlen_padded_i - half * (cp_rank + 1)
            back_end = src_offset + seqlen_padded_i - half * cp_rank
            out[dst_offset + half : dst_offset + seqlen_per_cp] = packed[back_start:back_end]
            src_offset += seqlen_padded_i
            dst_offset += seqlen_per_cp
        packed = out

    return packed.unsqueeze(0)  # [1, packed_len_per_cp, layers, topk]


def _get_current_pp_stage_layer_range(model_config) -> tuple[int, int]:
    """Return the current PP rank's transformer-layer range as (start_layer,
    num_layers).

    Prefer Megatron's own helpers so replay indexing stays aligned with the
    actual model partition, including embedding/loss pipeline accounting.
    """
    import megatron.core.parallel_state as mpu
    from megatron.core.transformer.transformer_block import get_num_layers_to_build
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

    pp_rank = mpu.get_pipeline_model_parallel_rank()
    offset = get_transformer_layer_offset(model_config, pp_rank=pp_rank)
    num_layers = get_num_layers_to_build(model_config, pp_rank=pp_rank)
    return offset, num_layers


def setup_per_microbatch_replay_forward(
    rollout_expert_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    model_config,
    use_sample_packing: bool = False,
) -> None:
    """Set up RouterReplay for a single micro-batch, aligning indices
    with the left-padding-removed token layout that the MoE layer sees.

    Handles context parallelism: when CP > 1, the sequence is split into
    2*cp_size chunks with each CP rank receiving a front chunk and a back
    chunk (for causal-mask load balancing). Replay indices are split using
    the same pattern so they stay aligned with the tokens each rank sees.

    Handles sequence parallelism: when TP > 1, the sequence is split across
    TP ranks, so each rank's MoE router only sees its local chunk of tokens.

    Handles dense-layer mismatch: DeepSeek V3-style models have dense FFN
    layers before the MoE layers. vLLM reports routing indices for ALL
    transformer layers, but Megatron only has RouterReplay instances for MoE
    layers. We use each instance's global layer_number (set by the patched
    TopKRouter.set_layer_number) to index into the correct slice of the data.

    Handles pipeline parallelism: when PP > 1, the sequence is split across
    PP ranks, so each rank only sees its local RouterReplay instances. In cases
    where the number of local RouterReplay instances does not match the local
    layer count, indicating that the model has dense layers before MoE layers,
    we use the global layer_number to index into the correct slice of the data.

    """
    import megatron.core.parallel_state as mpu
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    _patch_alltoall_dispatcher_for_replay()

    if use_sample_packing:
        aligned = _pack_replay_indices(rollout_expert_indices, attention_mask)
    else:
        aligned = _remove_left_padding_from_indices(rollout_expert_indices, attention_mask)

    # TP splitting: sequence parallelism across the tensor model parallel region
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size > 1:
        tp_rank = mpu.get_tensor_model_parallel_rank()
        seq_len = aligned.shape[1]
        chunk_size = seq_len // tp_size
        aligned = aligned[:, tp_rank * chunk_size : (tp_rank + 1) * chunk_size, :, :]
    per_layer_data = _split_replay_indices(aligned)
    global_num_layers_in_data = len(per_layer_data)
    instances = RouterReplay.global_router_replay_instances
    num_instances = len(instances)
    local_layer_offset, local_num_layers = _get_current_pp_stage_layer_range(model_config)

    if local_num_layers == num_instances:
        local_per_layer_data = per_layer_data[local_layer_offset : local_layer_offset + local_num_layers]
        RouterReplay.set_replay_data(local_per_layer_data)
    else:
        # Dense-layer mismatch: map each MoE router to its global layer index.
        # Prefer the patched layer_number; fall back to offset-based mapping
        # (assumes dense layers precede MoE layers).
        for local_router_idx, router_instance in enumerate(instances):
            layer_number = getattr(router_instance, "layer_number", None)
            if layer_number is not None:
                layer_idx = layer_number - 1  # layer_number is 1-based
            else:
                layer_idx = local_layer_offset + local_router_idx + (local_num_layers - num_instances)
            if layer_idx < 0 or layer_idx >= global_num_layers_in_data:
                raise ValueError(
                    f"Router replay layer index {layer_idx} out of range "
                    f"for data with {global_num_layers_in_data} layers "
                    f"({num_instances} router instances)"
                )
            router_instance.set_target_indices(per_layer_data[layer_idx])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)


def setup_per_microbatch_replay_backward() -> None:
    """Switch RouterReplay to backward mode so that activation-checkpoint
    recomputation during the backward pass consumes indices from
    ``replay_backward_list`` in FIFO order (populated during the forward pass).
    """
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)


def clear_router_replay():
    """Clear all router replay state."""
    from megatron.core.transformer.moe.router_replay import RouterReplay

    RouterReplay.clear_global_indices()
    RouterReplay.clear_global_router_replay_action()
