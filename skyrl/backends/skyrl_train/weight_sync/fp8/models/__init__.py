"""Per-model quantization specs for serialized FP8 weight sync."""

from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import (
    ModelFp8Spec,
    MoeExpertSpec,
    MoeProjection,
    batched_moe_wire_targets,
    engine_exclude_list,
    register_fp8_spec,
    registered_fp8_spec_names,
    resolve_fp8_spec,
    resolve_user_provided_exclude_list,
)

# Importing a model module registers its spec.
from skyrl.backends.skyrl_train.weight_sync.fp8.models.glm5 import GLM5_FP8_SPEC
from skyrl.backends.skyrl_train.weight_sync.fp8.models.qwen35 import QWEN35_FP8_SPEC

__all__ = [
    "GLM5_FP8_SPEC",
    "ModelFp8Spec",
    "MoeExpertSpec",
    "MoeProjection",
    "QWEN35_FP8_SPEC",
    "batched_moe_wire_targets",
    "engine_exclude_list",
    "register_fp8_spec",
    "registered_fp8_spec_names",
    "resolve_fp8_spec",
    "resolve_user_provided_exclude_list",
]
