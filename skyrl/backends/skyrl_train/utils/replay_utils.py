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
    sp_world_size = mpu.get_tensor_model_parallel_world_size()
    if sp_world_size > 1:
        pad_size = (sp_world_size - effective_seq_len % sp_world_size) % sp_world_size
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
    if cp_size > 1:
        total_packed_len = total_packed_len // cp_size

    packed = torch.zeros(
        total_packed_len,
        num_layers,
        topk,
        dtype=rollout_expert_indices.dtype,
        device=rollout_expert_indices.device,
    )

    seq_lens_cpu = seq_lens.tolist()
    seqlens_padded_cpu = seqlens_padded.tolist()
    if cp_size > 1:
        cp_rank = mpu.get_context_parallel_rank()
    offset = 0
    for i in range(batch_size):
        n = seq_lens_cpu[i]
        mask = attention_mask[i].bool()
        d = rollout_expert_indices[i, mask]
        if cp_size > 1:
            chunk_size = seqlens_padded_cpu[i] // cp_size
            start = cp_rank * chunk_size
            end = min(start + chunk_size, n)
            valid_len = max(0, end - start)
            if valid_len > 0:
                packed[offset : offset + valid_len] = d[start:end]
            offset += chunk_size
        else:
            packed[offset : offset + n] = d
            offset += seqlens_padded_cpu[i]

    return packed.unsqueeze(0)  # [1, total_packed_len, layers, topk]


def setup_per_microbatch_replay_forward(
    rollout_expert_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    use_sample_packing: bool = False,
) -> None:
    """Set up RouterReplay for a single micro-batch, aligning indices
    with the token layout that the MoE layer sees.

    Handles sequence parallelism: when TP > 1, the sequence is split across
    TP ranks, so each rank's MoE router only sees its local chunk of tokens.

    Handles sample packing: when use_sample_packing is True, sequences are
    concatenated into one packed sequence with per-sample alignment padding.
    The replay indices must follow this same packed layout.

    Handles dense-layer mismatch: DeepSeek V3-style models have dense FFN
    layers before the MoE layers.  vLLM reports routing indices for ALL
    transformer layers, but Megatron only has RouterReplay instances for MoE
    layers.  We use each instance's global layer_number (set by the patched
    TopKRouter.set_layer_number) to index into the correct slice of the data.
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

    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size > 1:
        tp_rank = mpu.get_tensor_model_parallel_rank()
        seq_len = aligned.shape[1]
        chunk_size = seq_len // tp_size
        aligned = aligned[:, tp_rank * chunk_size : (tp_rank + 1) * chunk_size, :, :]

    per_layer_data = _split_replay_indices(aligned)
    num_layers_in_data = len(per_layer_data)
    instances = RouterReplay.global_router_replay_instances
    num_instances = len(instances)

    if num_layers_in_data == num_instances:
        RouterReplay.set_replay_data(per_layer_data)
    else:
        # Dense-layer mismatch: map each MoE router to its global layer index.
        # Prefer the patched layer_number; fall back to offset-based mapping
        # (assumes dense layers precede MoE layers).
        for i, router_instance in enumerate(instances):
            layer_number = getattr(router_instance, "layer_number", None)
            if layer_number is not None:
                layer_idx = layer_number - 1  # layer_number is 1-based
            else:
                layer_idx = i + (num_layers_in_data - num_instances)
            if layer_idx < 0 or layer_idx >= num_layers_in_data:
                raise ValueError(
                    f"Router replay layer index {layer_idx} out of range "
                    f"for data with {num_layers_in_data} layers "
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
