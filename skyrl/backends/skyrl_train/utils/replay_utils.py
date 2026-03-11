"""
Utility functions for MoE Router Replay.
"""

import torch
from typing import List


def _patch_topk_router_layer_number():
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
        from megatron.core.transformer.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
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


def _get_current_pp_stage_layer_range(model_config) -> tuple[int, int]:
    """Return the current PP rank's transformer-layer range.

    Prefer Megatron's own helpers so replay indexing stays aligned with the
    actual model partition, including embedding/loss pipeline accounting.
    """
    import megatron.core.parallel_state as mpu
    from megatron.core.transformer.transformer_layer import get_transformer_layer_offset
    from megatron.core.transformer.transformer_block import get_num_layers_to_build


    if get_num_layers_to_build is not None:
        return get_transformer_layer_offset(model_config), get_num_layers_to_build(model_config, pp_rank=pp_rank)

    pp_size = mpu.get_pipeline_model_parallel_world_size()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    total_layers = model_config.num_layers
    first_stage_layers = getattr(model_config, "num_layers_in_first_pipeline_stage", None)
    last_stage_layers = getattr(model_config, "num_layers_in_last_pipeline_stage", None)

    if pp_size <= 1:
        return 0, total_layers

    if first_stage_layers is None and last_stage_layers is None:
        assert total_layers % pp_size == 0, (
            "For even pipelineing, num_layers should be divisible by pipeline_model_parallel_size"
        )
        pp_layers = total_layers // pp_size
        return pp_rank * pp_layers, pp_layers

    next_n_pp_layers = total_layers
    next_n_pp_stages = pp_size

    if first_stage_layers is not None:
        next_n_pp_layers -= first_stage_layers
        next_n_pp_stages -= 1

    if last_stage_layers is not None:
        next_n_pp_layers -= last_stage_layers
        next_n_pp_stages -= 1

    if next_n_pp_stages > 0:
        assert next_n_pp_layers % next_n_pp_stages == 0, (
            "Uneven pipelineing, not divisible by remaining pipeline stages"
        )
        next_n_pp_layers = next_n_pp_layers // next_n_pp_stages
    else:
        next_n_pp_layers = 0

    if pp_rank == 0 and first_stage_layers is not None:
        return 0, first_stage_layers

    if pp_rank == pp_size - 1 and last_stage_layers is not None:
        if first_stage_layers is not None:
            start = first_stage_layers + (next_n_pp_layers * (pp_size - 2))
        else:
            start = next_n_pp_layers * (pp_size - 1)
        return start, last_stage_layers

    if first_stage_layers is not None:
        return first_stage_layers + (next_n_pp_layers * (pp_rank - 1)), next_n_pp_layers
    return next_n_pp_layers * pp_rank, next_n_pp_layers


def setup_per_microbatch_replay_forward(
    rollout_expert_indices: torch.Tensor,
    attention_mask: torch.Tensor,
    model_config,
) -> None:
    """Set up RouterReplay for a single micro-batch, aligning indices
    with the left-padding-removed token layout that the MoE layer sees.

    Handles sequence parallelism: when TP > 1, the sequence is split across
    TP ranks, so each rank's MoE router only sees its local chunk of tokens.

    Handles dense-layer mismatch: DeepSeek V3-style models have dense FFN
    layers before the MoE layers.  vLLM reports routing indices for ALL
    transformer layers, but Megatron only has RouterReplay instances for MoE
    layers.  We use each instance's global layer_number (set by the patched
    TopKRouter.set_layer_number) to index into the correct slice of the data.
    """
    import megatron.core.parallel_state as mpu
    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

    _patch_alltoall_dispatcher_for_replay()

    aligned = _remove_left_padding_from_indices(rollout_expert_indices, attention_mask)

    # handles megatron sequence parallelism across the tensor model parallel region
    # since we automatically enable sequence parallelism when TP > 1
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
        for i, router_instance in enumerate(instances):
            layer_number = getattr(router_instance, "layer_number", None)
            if layer_number is not None:
                layer_idx = layer_number - 1  # layer_number is 1-based
            else:
                layer_idx = local_layer_offset + i
            if layer_idx < 0 or layer_idx >= global_num_layers_in_data:
                raise ValueError(
                    f"Router replay layer index {layer_idx} out of range "
                    f"for data with {global_num_layers_in_data} layers "
                    f"({num_instances} router instances)"
                )
            router_instance.set_target_indices(per_layer_data[layer_idx])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)


def clear_router_replay():
    """Clear all router replay state."""
    from megatron.core.transformer.moe.router_replay import RouterReplay

    RouterReplay.clear_global_indices()
    RouterReplay.clear_global_router_replay_action()
    RouterReplay.clear_global_router_replay_instances()
