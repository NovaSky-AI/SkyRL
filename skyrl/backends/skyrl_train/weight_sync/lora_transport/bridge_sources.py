"""Normalize Megatron-Bridge rank-local adapter records for LoRA transport."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Literal, Mapping

import torch


@dataclass(frozen=True)
class LoRABridgeSource:
    """Content-independent metadata for one rank-local Bridge adapter source."""

    key: str
    source_rank: int
    hf_param_names: tuple[str, ...]
    component: Literal["linear_in", "linear_out"]
    transform: Literal[
        "identity", "replicate", "split_qkv", "split_gated_mlp", "split_gdn_in_proj"
    ]
    shape: tuple[int, ...]
    tensor_parallel_axis: int | None
    tensor_parallel_rank: int
    tensor_parallel_size: int
    expert_parallel_axis: int | None
    expert_parallel_rank: int
    expert_parallel_size: int
    transform_config: tuple[tuple[str, int | bool | None], ...]


@dataclass(frozen=True)
class LoRABridgeSourceLayout:
    """Fixed rank-local Bridge source layout for one named LoRA adapter."""

    adapter_name: str
    sources: tuple[LoRABridgeSource, ...]
    source_dtype: str = "float32"
    layout_digest: str = field(init=False)

    def to_json_dict(self) -> dict[str, Any]:
        """Return JSON-safe control-plane metadata without adapter values."""
        return {
            "adapter_name": self.adapter_name,
            "source_dtype": self.source_dtype,
            "layout_digest": self.layout_digest,
            "sources": [asdict(source) for source in self.sources],
        }

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> "LoRABridgeSourceLayout":
        """Reconstruct and verify immutable layout metadata from the control plane."""
        source_layout = cls(
            adapter_name=data["adapter_name"],
            source_dtype=data["source_dtype"],
            sources=tuple(
                LoRABridgeSource(
                    key=source["key"],
                    source_rank=source["source_rank"],
                    hf_param_names=tuple(source["hf_param_names"]),
                    component=source["component"],
                    transform=source["transform"],
                    shape=tuple(source["shape"]),
                    tensor_parallel_axis=source["tensor_parallel_axis"],
                    tensor_parallel_rank=source["tensor_parallel_rank"],
                    tensor_parallel_size=source["tensor_parallel_size"],
                    expert_parallel_axis=source["expert_parallel_axis"],
                    expert_parallel_rank=source["expert_parallel_rank"],
                    expert_parallel_size=source["expert_parallel_size"],
                    transform_config=tuple(
                        (name, value) for name, value in source["transform_config"]
                    ),
                )
                for source in data["sources"]
            ),
        )
        if data["layout_digest"] != source_layout.layout_digest:
            raise ValueError("LoRA Bridge layout digest does not match its metadata")
        return source_layout

    def __post_init__(self) -> None:
        if not self.adapter_name:
            raise ValueError("LoRA Bridge layouts require a non-empty adapter name")
        if self.source_dtype != "float32":
            raise ValueError(
                f"lora_transport requires float32 Bridge sources, got {self.source_dtype!r}"
            )
        sources = tuple(sorted(self.sources, key=_bridge_source_sort_key))
        if sources != self.sources:
            raise ValueError("LoRA Bridge layout sources must be in canonical order")
        validate_lora_bridge_source_layout(sources)
        _validate_complete_lora_bridge_source_layout(sources)
        payload = {
            "adapter_name": self.adapter_name,
            "source_dtype": self.source_dtype,
            "sources": [asdict(source) for source in sources],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        object.__setattr__(self, "layout_digest", hashlib.sha256(encoded).hexdigest())


def extract_lora_bridge_sources(
    records: Iterable[Any],
    source_rank: int = 0,
) -> tuple[dict[str, torch.Tensor], tuple[LoRABridgeSource, ...]]:
    """Detach tensors from Bridge records and return stable transport metadata.

    The Bridge extension supplies a local FP32 snapshot per source tensor. This
    function keeps that storage separate from its serializable layout metadata so
    a producer can expose only the tensors over a native transport and publish the metadata on
    the control plane.
    """
    if source_rank < 0:
        raise ValueError(
            f"lora_transport Bridge source rank must be non-negative, got {source_rank}"
        )
    tensors: dict[str, torch.Tensor] = {}
    sources: list[LoRABridgeSource] = []
    for record in records:
        key = record.global_param_name
        if key in tensors:
            raise ValueError(f"Bridge adapter records contain duplicate source {key!r}")
        tensor = record.weight
        if tensor.dtype is not torch.float32:
            raise ValueError(
                f"lora_transport requires float32 Bridge source {key!r}, got {tensor.dtype}"
            )
        if not tensor.is_contiguous():
            raise ValueError(
                f"lora_transport requires contiguous Bridge source {key!r}"
            )
        if not record.hf_param_names:
            raise ValueError(f"Bridge adapter source {key!r} has no HF parameter names")
        tensors[key] = tensor
        sources.append(
            LoRABridgeSource(
                key=key,
                source_rank=source_rank,
                hf_param_names=tuple(record.hf_param_names),
                component=record.component,
                transform=record.transform,
                shape=tuple(tensor.shape),
                tensor_parallel_axis=record.tensor_parallel_axis,
                tensor_parallel_rank=record.tensor_parallel_rank,
                tensor_parallel_size=record.tensor_parallel_size,
                expert_parallel_axis=record.expert_parallel_axis,
                expert_parallel_rank=record.expert_parallel_rank,
                expert_parallel_size=record.expert_parallel_size,
                transform_config=tuple(record.transform_config),
            )
        )
    if not sources:
        raise ValueError("lora_transport requires at least one Bridge adapter source")
    sources.sort(key=lambda source: source.key)
    return tensors, tuple(sources)


def validate_lora_bridge_source_layout(
    sources: Iterable[LoRABridgeSource],
) -> Mapping[tuple[str, int, int], LoRABridgeSource]:
    """Validate a fixed Bridge source layout before publishing any generation."""
    source_tuple = tuple(sources)
    if not source_tuple:
        raise ValueError("lora_transport requires at least one Bridge adapter source")
    layout = {
        (source.key, source.tensor_parallel_rank, source.expert_parallel_rank): source
        for source in source_tuple
    }
    if len(layout) != len(source_tuple):
        raise ValueError("lora_transport Bridge source shard ownership must be unique")
    for source in source_tuple:
        _validate_lora_bridge_source(source)
    return layout


def _validate_lora_bridge_source(source: LoRABridgeSource) -> None:
    """Validate metadata that is meaningful for an individual local shard."""
    if source.source_rank < 0:
        raise ValueError(f"Bridge source {source.key!r} has invalid source rank")
    if any(dimension <= 0 for dimension in source.shape):
        raise ValueError(
            f"Bridge source {source.key!r} has invalid shape {source.shape!r}"
        )
    if source.tensor_parallel_size <= 0 or source.expert_parallel_size <= 0:
        raise ValueError(f"Bridge source {source.key!r} has invalid parallel sizes")
    if not 0 <= source.tensor_parallel_rank < source.tensor_parallel_size:
        raise ValueError(
            f"Bridge source {source.key!r} has invalid tensor-parallel rank"
        )
    if not 0 <= source.expert_parallel_rank < source.expert_parallel_size:
        raise ValueError(
            f"Bridge source {source.key!r} has invalid expert-parallel rank"
        )


def _validate_complete_lora_bridge_source_layout(
    sources: Iterable[LoRABridgeSource],
) -> None:
    """Require every TP/EP shard before a receiver can accept the layout."""
    grouped: dict[str, list[LoRABridgeSource]] = {}
    for source in sources:
        grouped.setdefault(source.key, []).append(source)
    for key, group in grouped.items():
        first = group[0]
        if any(
            source.hf_param_names != first.hf_param_names
            or source.component != first.component
            or source.transform != first.transform
            or source.tensor_parallel_axis != first.tensor_parallel_axis
            or source.tensor_parallel_size != first.tensor_parallel_size
            or source.expert_parallel_axis != first.expert_parallel_axis
            or source.expert_parallel_size != first.expert_parallel_size
            or source.transform_config != first.transform_config
            for source in group
        ):
            raise ValueError(f"Bridge source {key!r} has inconsistent shard metadata")
        expected = {
            (tensor_parallel_rank, expert_parallel_rank)
            for tensor_parallel_rank in range(first.tensor_parallel_size)
            for expert_parallel_rank in range(first.expert_parallel_size)
        }
        actual = {
            (source.tensor_parallel_rank, source.expert_parallel_rank)
            for source in group
        }
        if actual != expected:
            raise ValueError(f"Bridge source {key!r} is missing shard ownership")


def _bridge_source_sort_key(source: LoRABridgeSource) -> tuple[str, int, int, int]:
    return (
        source.key,
        source.expert_parallel_rank,
        source.tensor_parallel_rank,
        source.source_rank,
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
            ep_sources = [
                source for source in group if source.expert_parallel_rank == ep_rank
            ]
            if not ep_sources:
                raise ValueError(f"Bridge source {key!r} is missing EP rank {ep_rank}")
            shards = []
            expected_tp_ranks = range(first.tensor_parallel_size)
            for tp_rank in expected_tp_ranks:
                matching = [
                    source
                    for source in ep_sources
                    if source.tensor_parallel_rank == tp_rank
                ]
                if len(matching) != 1:
                    raise ValueError(
                        f"Bridge source {key!r} has invalid TP ownership for rank {tp_rank}"
                    )
                source = matching[0]
                tensor_key = (key, tp_rank, ep_rank)
                tensor = tensors.get(tensor_key)
                if tensor is None:
                    raise ValueError(
                        f"Bridge source {key!r} is missing pulled tensor {tensor_key!r}"
                    )
                if tensor.dtype is not torch.float32:
                    raise ValueError(
                        f"lora_transport requires float32 Bridge source {key!r}, got {tensor.dtype}"
                    )
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
            raise ValueError(
                f"Bridge source {source.key!r} identity transform requires one HF name"
            )
        result[source.hf_param_names[0]] = tensor
        return
    if source.transform == "replicate":
        for name in source.hf_param_names:
            result[name] = tensor
        return
    if source.transform == "split_gated_mlp":
        if len(source.hf_param_names) != 2:
            raise ValueError(
                f"Bridge source {source.key!r} gated transform requires two HF names"
            )
        gate, up = torch.chunk(tensor, 2, dim=0)
        result[source.hf_param_names[0]] = gate
        result[source.hf_param_names[1]] = up
        return
    if source.transform == "split_qkv":
        if len(source.hf_param_names) != 3:
            raise ValueError(
                f"Bridge source {source.key!r} QKV transform requires three HF names"
            )
        q, k, v = _split_qkv_lora_tensor(tensor, dict(source.transform_config))
        result[source.hf_param_names[0]] = q
        result[source.hf_param_names[1]] = k
        result[source.hf_param_names[2]] = v
        return
    if source.transform == "split_gdn_in_proj":
        if len(source.hf_param_names) != 4:
            raise ValueError(
                f"Bridge source {source.key!r} GDN transform requires four HF names"
            )
        parts = _split_gdn_lora_tensor(
            tensor,
            dict(source.transform_config),
            source.tensor_parallel_size,
        )
        for name, part in zip(source.hf_param_names, parts, strict=True):
            result[name] = part
        return
    raise ValueError(
        f"Bridge source {source.key!r} requires {source.transform!r} conversion with the Megatron config"
    )


def _split_qkv_lora_tensor(
    tensor: torch.Tensor,
    config: Mapping[str, int | bool | None],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split a full Megatron interleaved QKV LoRA-B tensor into PEFT tensors."""
    head_size, indices = get_qkv_lora_head_mapping(config)
    qkv_total_dim = sum(len(output_indices) for output_indices in indices)
    if tensor.ndim != 2 or tensor.shape[0] != qkv_total_dim * head_size:
        raise ValueError(
            f"QKV LoRA source has shape {tuple(tensor.shape)}, expected first dimension "
            f"{qkv_total_dim * head_size}"
        )
    feature_dim = tensor.shape[1]
    qkv = tensor.view(qkv_total_dim, head_size, feature_dim)
    return tuple(
        qkv[list(output_indices)].reshape(-1, feature_dim) for output_indices in indices
    )


def get_qkv_lora_head_mapping(
    config: Mapping[str, int | bool | None],
) -> tuple[int, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    """Return head size and source-head order for canonical Q, K, and V factors."""
    required = ("num_attention_heads", "num_query_groups", "kv_channels", "hidden_size")
    if any(config.get(field) is None for field in required):
        raise ValueError("QKV LoRA source is missing its Bridge transform config")
    num_heads = int(config["num_attention_heads"])
    num_groups = int(config["num_query_groups"])
    if num_heads <= 0 or num_groups <= 0 or num_heads % num_groups:
        raise ValueError(
            "QKV LoRA source requires evenly grouped positive attention heads"
        )
    head_size = int(config["kv_channels"] or int(config["hidden_size"]) // num_heads)
    if head_size <= 0:
        raise ValueError("QKV LoRA source requires a positive head size")
    heads_per_group = num_heads // num_groups
    attention_output_gate = bool(config.get("attention_output_gate", False))
    total_heads_per_group = (
        2 * heads_per_group + 2 if attention_output_gate else heads_per_group + 2
    )
    q_indices = tuple(
        total_heads_per_group * group + head
        for group in range(num_groups)
        for head in range(heads_per_group)
    )
    k_indices = tuple(
        total_heads_per_group * group + total_heads_per_group - 2
        for group in range(num_groups)
    )
    v_indices = tuple(
        total_heads_per_group * group + total_heads_per_group - 1
        for group in range(num_groups)
    )
    if attention_output_gate:
        z_indices = tuple(
            total_heads_per_group * group + heads_per_group + head
            for group in range(num_groups)
            for head in range(heads_per_group)
        )
        q_indices = tuple(
            index for pair in zip(q_indices, z_indices, strict=True) for index in pair
        )
    return head_size, (q_indices, k_indices, v_indices)


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
        raise ValueError(
            "GDN LoRA source head counts are not divisible by tensor parallel size"
        )
    feature_dim = tensor.shape[-1]
    qk_local = qk_head_dim * (num_qk_heads // tensor_parallel_size)
    v_local = v_head_dim * (num_v_heads // tensor_parallel_size)
    v_heads_local = num_v_heads // tensor_parallel_size
    rows_per_rank = 2 * qk_local + 2 * v_local + 2 * v_heads_local
    if tensor.ndim != 2 or tensor.shape[0] != tensor_parallel_size * rows_per_rank:
        raise ValueError(
            "GDN LoRA source shape does not match its Bridge transform config"
        )
    packed = tensor.reshape(tensor_parallel_size, rows_per_rank, feature_dim)
    q, k, v, z, b, a = torch.split(
        packed,
        [qk_local, qk_local, v_local, v_local, v_heads_local, v_heads_local],
        dim=1,
    )
    q, k, v, z, b, a = [
        part.reshape(num_qk_heads, -1, feature_dim) for part in (q, k, v, z, b, a)
    ]
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
