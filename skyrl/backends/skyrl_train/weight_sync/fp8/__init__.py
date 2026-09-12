"""Serialized FP8 weight sync: quantization, wire formats (blockwise + MXFP8), model specs."""

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
    MXFP8_GROUP_SIZE,
    batched_blockwise_cast_to_fp8,
    batched_mx_cast_to_fp8,
    blockwise_cast_to_fp8,
    mx_cast_to_fp8,
    normalize_block_size,
    use_power_2_scales_default,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.vllm_format import (
    AUTO_FP8,
    BLOCKWISE_FP8,
    MXFP8,
    SKYRL_BATCHED_MOE_FP8_PREFIX,
    WIRE_FORMATS,
    SerializedFp8Config,
    get_serialized_fp8_quantization_config,
    iter_batched_moe_expert_fp8_tensors,
    iter_serialized_fp8_tensors,
    scale_name_for_weight,
)

__all__ = [
    "AUTO_FP8",
    "BLOCKWISE_FP8",
    "MXFP8",
    "MXFP8_GROUP_SIZE",
    "WIRE_FORMATS",
    "SKYRL_BATCHED_MOE_FP8_PREFIX",
    "ModelFp8Spec",
    "MoeExpertSpec",
    "MoeProjection",
    "SerializedFp8Config",
    "batched_blockwise_cast_to_fp8",
    "batched_moe_wire_targets",
    "batched_mx_cast_to_fp8",
    "blockwise_cast_to_fp8",
    "mx_cast_to_fp8",
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
