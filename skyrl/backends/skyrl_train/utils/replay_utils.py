"""
Utility functions for MoE Router Replay.
"""

import torch
from typing import List
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch


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
    # try:
    #     from megatron.core.transformer.moe.router import TopKRouter
    # except ImportError:
    #     return

    # if getattr(TopKRouter, "_set_layer_number_patched", False):
    #     return

    # original_init = TopKRouter.__init__
    # original_set_layer_number = TopKRouter.set_layer_number

    # def patched_set_layer_number(self, layer_number: int):
    #     original_set_layer_number(self, layer_number)
    #     if self.router_replay is not None:
    #         self.router_replay.layer_number = layer_number

    # def patched_init(self, *args, **kwargs):
    #     # set_layer_num might not be regsitering before self.router_replay creation, enforce explicitly
    #     original_init(self, *args, **kwargs)
    #     if (self.router_replay is not None and hasattr(self, "layer_number") and self.layer_number is not None):
    #         self.router_replay.layer_number = self.layer_number

    # TopKRouter.__init__ = patched_init
    # TopKRouter.set_layer_number = patched_set_layer_number
    # TopKRouter._set_layer_number_patched = True

    try:
        from megatron.core.transformer.moe.router import TopKRouter
    except ImportError:
        return

    if getattr(TopKRouter, "_set_layer_number_patched", False):
        return

    original_set_layer_number = TopKRouter.set_layer_number

    def patched_set_layer_number(self, layer_number: int):
        original_set_layer_number(self, layer_number)
        if getattr(self, "router_replay", None) is not None:
            self.router_replay.layer_number = layer_number

    TopKRouter.set_layer_number = patched_set_layer_number
    TopKRouter._set_layer_number_patched = True


def _patch_alltoall_dispatcher_for_replay():
    """Monkey-patch MoEAlltoAllTokenDispatcher.preprocess to handle router replay.

    When router replay is enabled, duplicate indices in top_indices can cause
    routing_map.sum() < num_tokens * topk, leading to a split size mismatch
    in the alltoall collective.  We fix this by deriving num_out_tokens from
    the routing map instead of the static num_tokens * topk formula.
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


def filter_all_zero_layers_from_rii(rollout_inference_indices: torch.Tensor) -> torch.Tensor:
    if rollout_inference_indices is None or rollout_inference_indices.dim() != 4:
        return rollout_inference_indices
    num_layers = rollout_inference_indices.shape[2]
    first_moe_layer = 0
    for layer_idx in range(num_layers):
        layer_slice = rollout_inference_indices[:, :, layer_idx, :]
        if (layer_slice != 0).any().item():
            first_moe_layer = layer_idx
            break
    if first_moe_layer == 0:
        return rollout_inference_indices
    return rollout_inference_indices[:, :, first_moe_layer:, :].contiguous()


def _split_replay_indices(rollout_inference_indices: torch.Tensor) -> List[torch.Tensor]:
    if rollout_inference_indices is None:
        return None
    if rollout_inference_indices.dim() != 4:
        raise ValueError(f"Expected 4D replay indices, got shape {rollout_inference_indices.shape}")
    per_layer = rollout_inference_indices.permute(2, 0, 1, 3).contiguous()
    # flatten [batch, seq, topk] to [batch * seq, topk] for each layer
    return [per_layer[i].reshape(-1, per_layer.shape[-1]) for i in range(per_layer.shape[0])]


def _remove_left_padding_from_indices(
    rollout_inference_indices: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Apply the same left-padding removal as remove_left_padding to routing indices.

    Args:
        rollout_inference_indices: [batch, padded_seq_len, layers, topk]
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

    batch_size = rollout_inference_indices.shape[0]
    new_rii = torch.zeros(
        batch_size,
        effective_seq_len,
        rollout_inference_indices.shape[2],
        rollout_inference_indices.shape[3],
        dtype=rollout_inference_indices.dtype,
        device=rollout_inference_indices.device,
    )
    for i in range(batch_size):
        mask = attention_mask[i].bool()
        new_rii[i, : seq_lens[i]] = rollout_inference_indices[i, mask]
    return new_rii


def _setup_per_microbatch_replay(
    rollout_inference_indices: torch.Tensor,
    attention_mask: torch.Tensor,
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
    import logging
    import megatron.core.parallel_state as mpu
    from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

    # Drop leading all-zero layers (dense layers in DeepSeek-style models)
    rollout_inference_indices = filter_all_zero_layers_from_rii(rollout_inference_indices)

    _patch_alltoall_dispatcher_for_replay()

    aligned = _remove_left_padding_from_indices(rollout_inference_indices, attention_mask)
    logger = logging.getLogger(__name__)
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
            layer_numbers = [getattr(inst, "layer_number", None) for inst in instances]
            logger.info(
                f"[RouterReplay] Dense-layer: data has {num_layers_in_data} layers, "
                f"{num_instances} router instances. layer_numbers={layer_numbers}"
            )
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


def setup_router_replay_forward(
    data: TrainingInputBatch,
    enable_router_replay: bool,
    use_sample_packing: bool = False,
) -> bool:
    """
    Set up router replay for forward pass (ref/policy inference).
    """
    if not enable_router_replay:
        return False

    rollout_inference_indices = data.get("rollout_inference_indices")
    if rollout_inference_indices is None:
        return False

    attention_mask = data.get("attention_mask")
    if attention_mask is not None:
        # Use the same dense-layer-aware replay setup as real forward path.
        _setup_per_microbatch_replay(rollout_inference_indices, attention_mask)
    else:
        from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

        RouterReplay.set_replay_data(_split_replay_indices(rollout_inference_indices))
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    return True


def setup_router_replay_backward(
    data: TrainingInputBatch,
    enable_router_replay: bool,
    use_sample_packing: bool = False,
) -> bool:
    """
    Set up router replay for training forward/backward pass.
    """
    if not enable_router_replay:
        return False

    rollout_inference_indices = data.get("rollout_inference_indices")
    if rollout_inference_indices is None:
        return False

    attention_mask = data.get("attention_mask")
    if attention_mask is not None:
        _setup_per_microbatch_replay(rollout_inference_indices, attention_mask)
    else:
        from megatron.core.transformer.moe.router_replay import RouterReplay, RouterReplayAction

        RouterReplay.set_replay_data(_split_replay_indices(rollout_inference_indices))
        # Use REPLAY_FORWARD - Megatron handles REPLAY_BACKWARD automatically
        RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    return True


def clear_router_replay():
    """Clear all router replay state."""
    from megatron.core.transformer.moe.router_replay import RouterReplay

    RouterReplay.clear_global_indices()
    RouterReplay.clear_global_router_replay_action()
    # RouterReplay.clear_global_router_replay_instances()
