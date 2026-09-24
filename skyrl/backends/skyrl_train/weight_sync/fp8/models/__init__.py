"""Per-model quantization specs for serialized FP8 weight sync."""

from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import (
    ModelFp8Spec,
    MoeExpertSpec,
    MoeProjection,
    batched_moe_wire_targets,
    register_fp8_spec,
    registered_fp8_spec_names,
    resolve_fp8_spec,
)

# Importing a model module registers its spec.
from skyrl.backends.skyrl_train.weight_sync.fp8.models.qwen35 import QWEN35_FP8_SPEC

__all__ = [
    "ModelFp8Spec",
    "MoeExpertSpec",
    "MoeProjection",
    "QWEN35_FP8_SPEC",
    "batched_moe_wire_targets",
    "register_fp8_spec",
    "registered_fp8_spec_names",
    "resolve_fp8_spec",
]
