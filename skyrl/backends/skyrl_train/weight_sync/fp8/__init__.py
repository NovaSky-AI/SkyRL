"""Serialized blockwise FP8 weight sync: quantization, wire format, model specs."""

from skyrl.backends.skyrl_train.weight_sync.fp8.models import (
    ModelFp8Spec,
    MoeExpertSpec,
    MoeProjection,
    batched_moe_wire_targets,
    register_fp8_spec,
    registered_fp8_spec_names,
    resolve_fp8_spec,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.quantize import (
    batched_blockwise_cast_to_fp8,
    blockwise_cast_to_fp8,
    normalize_block_size,
    use_power_2_scales_default,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.vllm_format import (
    BLOCKWISE_FP8,
    SKYRL_BATCHED_MOE_FP8_PREFIX,
    SerializedFp8Config,
    get_serialized_fp8_quantization_config,
    iter_batched_moe_expert_fp8_tensors,
    iter_serialized_fp8_tensors,
    scale_name_for_weight,
)

__all__ = [
    "BLOCKWISE_FP8",
    "SKYRL_BATCHED_MOE_FP8_PREFIX",
    "ModelFp8Spec",
    "MoeExpertSpec",
    "MoeProjection",
    "SerializedFp8Config",
    "batched_blockwise_cast_to_fp8",
    "batched_moe_wire_targets",
    "blockwise_cast_to_fp8",
    "get_serialized_fp8_quantization_config",
    "iter_batched_moe_expert_fp8_tensors",
    "iter_serialized_fp8_tensors",
    "normalize_block_size",
    "register_fp8_spec",
    "registered_fp8_spec_names",
    "resolve_fp8_spec",
    "scale_name_for_weight",
    "use_power_2_scales_default",
]
