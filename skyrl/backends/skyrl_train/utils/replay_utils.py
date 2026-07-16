"""
Utility functions for MoE Router Replay.
"""

from contextlib import contextmanager

import torch

from skyrl.utils.token_metadata import (
    TokenMetadataLayout,
    align_token_metadata,
)


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


def patch_topk_router_expert_bias_padding_mask():
    """Fix the token-mask broadcast in pinned Megatron's expert-bias accounting."""
    try:
        from megatron.core.transformer.moe.router import TopKRouter
    except ImportError:
        return

    if getattr(TopKRouter, "_expert_bias_padding_mask_patched", False):
        return

    original_apply_expert_bias = TopKRouter._apply_expert_bias

    def patched_apply_expert_bias(self, routing_map: torch.Tensor, padding_mask: torch.Tensor | None = None):
        # Megatron combines [tokens, experts] with a token-only mask.
        if padding_mask is not None and padding_mask.ndim == 1:
            padding_mask = padding_mask.unsqueeze(-1)
        return original_apply_expert_bias(self, routing_map, padding_mask)

    TopKRouter._apply_expert_bias = patched_apply_expert_bias
    TopKRouter._expert_bias_padding_mask_patched = True


def _split_replay_indices(rollout_expert_indices: torch.Tensor) -> list[torch.Tensor]:
    per_layer = rollout_expert_indices.permute(2, 0, 1, 3).contiguous().to(torch.int32)
    return list(per_layer.flatten(1, 2).unbind(0))


def scatter_router_padding_mask_for_model(
    router_padding_mask: torch.Tensor | None,
    model,
    model_config,
) -> torch.Tensor | None:
    """Match the mask layout to sequence-parallel hidden states at model entry."""
    if router_padding_mask is None or not model_config.sequence_parallel:
        return router_padding_mask

    from megatron.core.models.hybrid.hybrid_model import HybridModel
    from megatron.core.tensor_parallel import scatter_to_sequence_parallel_region
    from megatron.core.utils import unwrap_model

    unwrapped_model = unwrap_model(model)
    # GPTModel scatters its mask beside the embedding on the first PP stage. HybridModel
    # scatters only the embedding, so its mask must always be scattered here.
    if not isinstance(unwrapped_model, HybridModel) and unwrapped_model.pre_process:
        return router_padding_mask
    return (
        scatter_to_sequence_parallel_region(router_padding_mask.transpose(0, 1).contiguous())
        .transpose(0, 1)
        .contiguous()
    )


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


def _get_local_router_layer_indices(model_config, global_num_layers: int, instances: list) -> list[int]:
    local_layer_offset, local_num_layers = _get_current_pp_stage_layer_range(model_config)
    if local_num_layers == len(instances):
        layer_indices = list(range(local_layer_offset, local_layer_offset + local_num_layers))
    else:
        layer_indices = []
        for local_router_index, router_instance in enumerate(instances):
            layer_number = getattr(router_instance, "layer_number", None)
            if layer_number is not None:
                layer_index = layer_number - 1
            else:
                layer_index = local_layer_offset + local_router_index + (local_num_layers - len(instances))
            layer_indices.append(layer_index)

    if any(layer_index < 0 or layer_index >= global_num_layers for layer_index in layer_indices):
        raise ValueError(
            f"Router replay layer indices {layer_indices} out of range for data with {global_num_layers} layers"
        )
    return layer_indices


def setup_per_microbatch_replay_forward(
    rollout_expert_indices: torch.Tensor,
    router_padding_mask: torch.Tensor | None,
    attention_mask: torch.Tensor,
    model,
    model_config,
    metadata_layout: TokenMetadataLayout,
    remove_microbatch_padding: bool = False,
) -> dict[str, torch.Tensor]:
    """Set up router replay and return its model-facing keyword arguments.

    Replay indices and the router padding mask start in the same batch layout and
    undergo matching padding removal or packing and CP sharding. Their destinations
    then differ: indices are TP-sliced and installed into per-layer ``RouterReplay``
    instances, while the mask follows Megatron's model-specific sequence-parallel
    path and is passed to the model as ``padding_mask``.

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

    Handles pipeline parallelism: when PP > 1, transformer layers are split
    across PP ranks, so each rank only sees its local RouterReplay instances. In cases
    where the number of local RouterReplay instances does not match the local
    layer count, indicating that the model has dense layers before MoE layers,
    we use the global layer_number to index into the correct slice of the data.

    """
    import megatron.core.parallel_state as mpu
    from megatron.core.transformer.moe.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )

    if router_padding_mask is None:
        raise ValueError("router_padding_mask is required with rollout_expert_indices")
    if rollout_expert_indices.dim() != 4:
        raise ValueError(f"Expected 4D replay indices, got shape {rollout_expert_indices.shape}")

    if router_padding_mask.shape != attention_mask.shape:
        raise ValueError(
            f"router_padding_mask shape {router_padding_mask.shape} does not match "
            f"attention_mask shape {attention_mask.shape}"
        )
    if router_padding_mask.device != rollout_expert_indices.device:
        raise ValueError("rollout_expert_indices and router_padding_mask must be on the same device")

    instances = RouterReplay.global_router_replay_instances
    local_layer_indices = _get_local_router_layer_indices(
        model_config,
        rollout_expert_indices.shape[2],
        instances,
    )
    layer_index = torch.tensor(local_layer_indices, dtype=torch.long, device=rollout_expert_indices.device)
    local_rollout_expert_indices = rollout_expert_indices.index_select(2, layer_index)

    if (metadata_layout.padded_sequence_lengths is not None) != remove_microbatch_padding:
        raise ValueError("Shared token metadata layout does not match the model packing mode")
    aligned_router_padding_mask = align_token_metadata(router_padding_mask.to(torch.bool), metadata_layout, True)
    route_padding = torch.arange(
        rollout_expert_indices.shape[-1],
        dtype=rollout_expert_indices.dtype,
        device=local_rollout_expert_indices.device,
    )
    aligned_rollout_expert_indices = align_token_metadata(
        local_rollout_expert_indices,
        metadata_layout,
        route_padding,
    )

    # TP splitting: sequence parallelism across the tensor model parallel region
    tp_size = mpu.get_tensor_model_parallel_world_size()
    if tp_size > 1:
        tp_rank = mpu.get_tensor_model_parallel_rank()
        seq_len = aligned_rollout_expert_indices.shape[1]
        chunk_size = seq_len // tp_size
        aligned_rollout_expert_indices = aligned_rollout_expert_indices[
            :, tp_rank * chunk_size : (tp_rank + 1) * chunk_size, :, :
        ]
    RouterReplay.set_replay_data(_split_replay_indices(aligned_rollout_expert_indices))
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    model_router_padding_mask = scatter_router_padding_mask_for_model(
        aligned_router_padding_mask,
        model,
        model_config,
    )
    return {"padding_mask": model_router_padding_mask}


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


@contextmanager
def router_replay_schedule(enabled: bool):
    """Isolate global RouterReplay state to one Megatron pipeline schedule.

    The backward FIFO spans all microbatches in a training schedule, so it must
    only be cleared at schedule boundaries. Forward-only schedules leave that
    FIFO unconsumed, and failed schedules may leave it partially consumed.
    """
    if not enabled:
        yield
        return

    clear_router_replay()
    try:
        yield
    finally:
        clear_router_replay()
