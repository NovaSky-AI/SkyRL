"""Full-tensor reference oracles for validating sliced LoRA transport."""

from typing import Iterable, Mapping

import torch

from skyrl.backends.skyrl_train.weight_sync.lora_transport.bridge_sources import (
    LoRABridgeSource,
    get_qkv_lora_head_mapping,
)
from skyrl.backends.skyrl_train.weight_sync.lora_transport.consumer_plan import (
    LoRAConsumerPlan,
    LoRAConsumerPull,
)


def reconstruct_lora_bridge_tensors(
    sources: Iterable[LoRABridgeSource],
    tensors: Mapping[tuple[str, int, int], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reconstruct PEFT tensors from pulled rank-local Bridge source shards.

    ``tensors`` keys are ``(global_param_name, tp_rank, ep_rank)``. The caller
    obtains those tensors through the selected transport; this function only performs deterministic
    local assembly using the immutable Bridge transformer configuration.
    """
    grouped: dict[str, list[LoRABridgeSource]] = {}
    for source in sources:
        grouped.setdefault(source.key, []).append(source)
    result: dict[str, torch.Tensor] = {}
    for key, group in grouped.items():
        first = group[0]
        if any(
            source.hf_param_names != first.hf_param_names
            or source.component != first.component
            or source.transform != first.transform
            or source.tensor_parallel_axis != first.tensor_parallel_axis
            or source.expert_parallel_axis != first.expert_parallel_axis
            or source.transform_config != first.transform_config
            for source in group
        ):
            raise ValueError(f"Bridge source {key!r} has inconsistent shard metadata")
        local_by_ep: list[torch.Tensor] = []
        for ep_rank in range(first.expert_parallel_size):
            ep_sources = [source for source in group if source.expert_parallel_rank == ep_rank]
            if not ep_sources:
                raise ValueError(f"Bridge source {key!r} is missing EP rank {ep_rank}")
            shards = []
            expected_tp_ranks = range(first.tensor_parallel_size)
            for tp_rank in expected_tp_ranks:
                matching = [source for source in ep_sources if source.tensor_parallel_rank == tp_rank]
                if len(matching) != 1:
                    raise ValueError(f"Bridge source {key!r} has invalid TP ownership for rank {tp_rank}")
                source = matching[0]
                tensor_key = (key, tp_rank, ep_rank)
                tensor = tensors.get(tensor_key)
                if tensor is None:
                    raise ValueError(f"Bridge source {key!r} is missing pulled tensor {tensor_key!r}")
                if tensor.dtype is not torch.float32:
                    raise ValueError(f"lora_transport requires float32 Bridge source {key!r}, got {tensor.dtype}")
                if tuple(tensor.shape) != source.shape:
                    raise ValueError(
                        f"Bridge source {key!r} tensor {tensor_key!r} has shape {tuple(tensor.shape)}, "
                        f"expected {source.shape}"
                    )
                shards.append(tensor)
            if first.tensor_parallel_axis is None:
                local_by_ep.append(shards[0])
            else:
                local_by_ep.append(torch.cat(shards, dim=first.tensor_parallel_axis))
        if first.expert_parallel_axis is None:
            assembled = local_by_ep[0]
        else:
            assembled = torch.cat(local_by_ep, dim=first.expert_parallel_axis)
        _emit_reconstructed_lora_tensors(result, first, assembled)
    return result


def _emit_reconstructed_lora_tensors(
    result: dict[str, torch.Tensor],
    source: LoRABridgeSource,
    tensor: torch.Tensor,
) -> None:
    """Apply a Bridge-declared post-assembly transform to one source tensor."""
    if source.transform == "identity":
        if len(source.hf_param_names) != 1:
            raise ValueError(f"Bridge source {source.key!r} identity transform requires one HF name")
        result[source.hf_param_names[0]] = tensor
        return
    if source.transform == "replicate":
        for name in source.hf_param_names:
            result[name] = tensor
        return
    if source.transform == "split_gated_mlp":
        if len(source.hf_param_names) != 2:
            raise ValueError(f"Bridge source {source.key!r} gated transform requires two HF names")
        gate, up = torch.chunk(tensor, 2, dim=0)
        result[source.hf_param_names[0]] = gate
        result[source.hf_param_names[1]] = up
        return
    if source.transform == "split_qkv":
        if len(source.hf_param_names) != 3:
            raise ValueError(f"Bridge source {source.key!r} QKV transform requires three HF names")
        q, k, v = _split_qkv_lora_tensor(tensor, dict(source.transform_config))
        result[source.hf_param_names[0]] = q
        result[source.hf_param_names[1]] = k
        result[source.hf_param_names[2]] = v
        return
    if source.transform == "split_gdn_in_proj":
        if len(source.hf_param_names) != 4:
            raise ValueError(f"Bridge source {source.key!r} GDN transform requires four HF names")
        parts = _split_gdn_lora_tensor(
            tensor,
            dict(source.transform_config),
            source.tensor_parallel_size,
        )
        for name, part in zip(source.hf_param_names, parts, strict=True):
            result[name] = part
        return
    raise ValueError(f"Bridge source {source.key!r} requires {source.transform!r} conversion with the Megatron config")


def _split_qkv_lora_tensor(
    tensor: torch.Tensor,
    config: Mapping[str, int | bool | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a full Megatron interleaved QKV LoRA-B tensor into PEFT tensors."""
    head_size, indices = get_qkv_lora_head_mapping(config)
    qkv_total_dim = sum(len(output_indices) for output_indices in indices)
    if tensor.ndim != 2 or tensor.shape[0] != qkv_total_dim * head_size:
        raise ValueError(
            f"QKV LoRA source has shape {tuple(tensor.shape)}, expected first dimension " f"{qkv_total_dim * head_size}"
        )
    feature_dim = tensor.shape[1]
    qkv = tensor.view(qkv_total_dim, head_size, feature_dim)
    return tuple(qkv[list(output_indices)].reshape(-1, feature_dim) for output_indices in indices)


def _split_gdn_lora_tensor(
    tensor: torch.Tensor,
    config: Mapping[str, int | bool | None],
    tensor_parallel_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a packed GLM DSA in-projection LoRA-B tensor into PEFT tensors."""
    fields = (
        "linear_key_head_dim",
        "linear_value_head_dim",
        "linear_num_key_heads",
        "linear_num_value_heads",
    )
    if any(config.get(field) is None for field in fields):
        raise ValueError("GDN LoRA source is missing its Bridge transform config")
    qk_head_dim = int(config["linear_key_head_dim"])
    v_head_dim = int(config["linear_value_head_dim"])
    num_qk_heads = int(config["linear_num_key_heads"])
    num_v_heads = int(config["linear_num_value_heads"])
    if num_qk_heads % tensor_parallel_size or num_v_heads % tensor_parallel_size:
        raise ValueError("GDN LoRA source head counts are not divisible by tensor parallel size")
    feature_dim = tensor.shape[-1]
    qk_local = qk_head_dim * (num_qk_heads // tensor_parallel_size)
    v_local = v_head_dim * (num_v_heads // tensor_parallel_size)
    v_heads_local = num_v_heads // tensor_parallel_size
    rows_per_rank = 2 * qk_local + 2 * v_local + 2 * v_heads_local
    if tensor.ndim != 2 or tensor.shape[0] != tensor_parallel_size * rows_per_rank:
        raise ValueError("GDN LoRA source shape does not match its Bridge transform config")
    packed = tensor.reshape(tensor_parallel_size, rows_per_rank, feature_dim)
    q, k, v, z, b, a = torch.split(
        packed,
        [qk_local, qk_local, v_local, v_local, v_heads_local, v_heads_local],
        dim=1,
    )
    q, k, v, z, b, a = [part.reshape(num_qk_heads, -1, feature_dim) for part in (q, k, v, z, b, a)]
    qkvz = torch.cat([q, k, v, z], dim=1)
    ba = torch.cat([b, a], dim=1)
    v_per_group = num_v_heads // num_qk_heads
    q_g, k_g, v_g, z_g = torch.split(
        qkvz,
        [qk_head_dim, qk_head_dim, v_per_group * v_head_dim, v_per_group * v_head_dim],
        dim=1,
    )
    b_g, a_g = torch.split(ba, [v_per_group, v_per_group], dim=1)
    qkv = torch.cat(
        [
            q_g.reshape(-1, feature_dim),
            k_g.reshape(-1, feature_dim),
            v_g.reshape(-1, feature_dim),
        ],
        dim=0,
    )
    return (
        qkv,
        z_g.reshape(-1, feature_dim),
        b_g.reshape(-1, feature_dim),
        a_g.reshape(-1, feature_dim),
    )


def assemble_lora_consumer_factors(
    plan: LoRAConsumerPlan,
    pulled: Mapping[LoRAConsumerPull, torch.Tensor],
    device: torch.device | str,
) -> dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]]:
    """Materialize independently scaled BF16 factors from exact FP32 pulls."""
    if set(pulled) != set(plan.pulls):
        raise ValueError("Pulled slices must match the complete consumer plan")
    for pull, tensor in pulled.items():
        shape = tuple(b - a for a, b in zip(pull.source_slice.starts, pull.source_slice.stops))
        if tensor.dtype != torch.float32 or tuple(tensor.shape) != shape:
            raise ValueError("Pulled slices must preserve exact FP32 shape and dtype")
    factors = {
        module.module_name: (
            [torch.zeros(pair[0], dtype=torch.bfloat16, device=device) for pair in module.factor_shapes],
            [torch.zeros(pair[1], dtype=torch.bfloat16, device=device) for pair in module.factor_shapes],
        )
        for module in plan.receiver_plan.modules
    }
    for copy in plan.copies:
        tensor = pulled[plan.pulls[copy.pull_index]]
        shape = tuple(b - a for a, b in zip(copy.starts, copy.stops))
        destination = factors[copy.module_name][copy.component][copy.factor_index]
        source = tensor.reshape(shape)
        if plan.pulls[copy.pull_index].value_scale != (1, 1):
            numerator, denominator = plan.pulls[copy.pull_index].value_scale
            source = source * (numerator / denominator)
        destination[tuple(slice(a, b) for a, b in zip(copy.starts, copy.stops))].copy_(source)
    return factors
