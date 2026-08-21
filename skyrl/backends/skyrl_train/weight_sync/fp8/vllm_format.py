"""Serialized wire formats for vLLM FP8 checkpoints: blockwise and MXFP8.

The two wires side by side — every branch in this package is one row of this
table, keyed off ``SerializedFp8Config.wire_format``:

                        blockwise                    mxfp8
  trainer recipe        Float8BlockScaling           MXFP8BlockScaling
  scale granularity     128x128 tiles                1x32 groups (along K)
  scale encoding        FP32 (or power-of-2)         E8M0 biased exponent, uint8
  scale tensor          .weight_scale_inv            .weight_scale
  vLLM quantization     fp8                          compressed-tensors
  cast kernels          blockwise_cast_to_fp8 /      mx_cast_to_fp8 /
                        batched_blockwise_cast...    batched_mx_cast_to_fp8

Sender (MegatronWeightExtractor) and receiver (worker-extension loaders) are
wire-format-agnostic; the format decides only what this serializer emits, the
quantization config injected at engine boot, and the per-model ignore lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Sequence

import torch

from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import (
    AUTO_FP8,
    BLOCKWISE_FP8,
    MXFP8,
    WIRE_FORMATS,
    WIRE_SCALE_SUFFIX,
    ModelFp8Spec,
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

__all__ = [  # re-exported so callers keep importing formats from the serializer
    "AUTO_FP8",
    "BLOCKWISE_FP8",
    "MXFP8",
    "WIRE_FORMATS",
]

# Internal wire-format marker for Qwen3.5 MoE tensors that remain batched over
# experts. The receiver strips this marker and routes the tensor directly to
# vLLM's fused-MoE parameter loader instead of the ordinary HF-name loader.
SKYRL_BATCHED_MOE_FP8_PREFIX = "__skyrl_batched_moe_fp8__:"


@dataclass(frozen=True)
class SerializedFp8Config:
    """Configuration for serialized FP8 rollout weight sync.

    ``spec`` is the per-model quantization policy, resolved once from the HF
    config via ``resolve_fp8_spec``; the tensor iterators require it.
    """

    # blockwise-wire parameters; the MXFP8 wire has no equivalents (its group
    # geometry and scale encoding are fixed by the OCP microscaling spec).
    weight_block_size: tuple[int, int] = (128, 128)
    power_2_scale: bool = field(default_factory=use_power_2_scales_default)
    # shared across wires
    spec: ModelFp8Spec | None = None
    wire_format: str = BLOCKWISE_FP8

    def __post_init__(self) -> None:
        object.__setattr__(self, "weight_block_size", normalize_block_size(self.weight_block_size))
        if type(self.power_2_scale) is not bool:
            raise ValueError(f"power_2_scale must be a bool, got {self.power_2_scale!r}")
        if self.wire_format not in WIRE_FORMATS:
            raise ValueError(f"wire_format must be one of {WIRE_FORMATS}, got {self.wire_format!r}")

    @property
    def is_mxfp8(self) -> bool:
        return self.wire_format == MXFP8

    def require_spec(self) -> ModelFp8Spec:
        if self.spec is None:
            raise ValueError(
                "SerializedFp8Config.spec is not set; resolve the model spec with "
                "resolve_fp8_spec(hf_config) before serializing weights"
            )
        return self.spec


def _mxfp8_group_args(dynamic: bool) -> dict:
    """One compressed-tensors group descriptor for MXFP8.

    vLLM selects its MXFP8 schemes -- dense and fused-MoE alike -- from this
    exact predicate (``CompressedTensorsConfig._is_mxfp8``): group strategy,
    symmetric, group size 32, float, 8 bits, and a uint8 scale dtype.
    """

    return {
        "num_bits": 8,
        "type": "float",
        "strategy": "group",
        "group_size": MXFP8_GROUP_SIZE,
        "symmetric": True,
        "scale_dtype": "uint8",
        "dynamic": dynamic,
    }


def get_serialized_fp8_quantization_config(
    weight_block_size: Sequence[int] = (128, 128),
    ignored_layers: Sequence[str] | None = None,
    wire_format: str = BLOCKWISE_FP8,
) -> dict:
    """Return vLLM's Hugging Face quantization config for serialized FP8."""

    if wire_format not in WIRE_FORMATS:
        raise ValueError(f"wire_format must be one of {WIRE_FORMATS}, got {wire_format!r}")

    if wire_format == MXFP8:
        # MXFP8 is served through compressed-tensors, which names the excluded
        # modules "ignore" rather than vLLM-fp8's "ignored_layers".
        return {
            "quant_method": "compressed-tensors",
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": _mxfp8_group_args(dynamic=False),
                    "input_activations": _mxfp8_group_args(dynamic=True),
                }
            },
            "ignore": list(ignored_layers or ()),
        }

    block_m, block_n = normalize_block_size(weight_block_size)
    qconfig = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [block_m, block_n],
    }
    if ignored_layers:
        qconfig["ignored_layers"] = list(ignored_layers)
    return qconfig


def scale_name_for_weight(name: str, wire_format: str = BLOCKWISE_FP8) -> str:
    """Return the scale tensor name paired with a quantized weight.

    Blockwise ships an inverse FP32 scale per 128x128 tile; MXFP8 ships an E8M0
    exponent per 32-element group, which compressed-tensors loads as
    ``weight_scale``.
    """

    if not name.endswith(".weight"):
        raise ValueError(f"FP8 scale can only be derived from .weight tensors: {name}")
    suffix = WIRE_SCALE_SUFFIX[MXFP8] if wire_format == MXFP8 else WIRE_SCALE_SUFFIX[BLOCKWISE_FP8]
    return name[: -len(".weight")] + suffix


def iter_batched_moe_expert_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Convert a batched expert tensor without expanding expert names.

    The old wire format emitted one weight and one scale tensor for every
    expert/projection pair. Keeping the expert dimension intact reduces each
    routed MoE layer from ``6 * num_experts`` tensors to six and lets vLLM use
    its fused 3D loader.
    """
    moe_spec = config.require_spec().moe_expert_spec(name)
    if moe_spec is None:
        raise ValueError(f"Not a batched MoE expert tensor: {name}")
    if tensor.ndim != 3:
        raise ValueError(f"Batched MoE expert tensor must be 3D, got shape={tuple(tensor.shape)}")
    if moe_spec.split_dim is not None:
        num_projections = len(moe_spec.projections)
        if tensor.shape[moe_spec.split_dim] % num_projections != 0:
            raise ValueError(
                f"Batched MoE tensor dim {moe_spec.split_dim} must split evenly across "
                f"{num_projections} projections, got shape={tuple(tensor.shape)}"
            )
        projection_tensors = torch.chunk(tensor, num_projections, dim=moe_spec.split_dim)
    else:
        projection_tensors = (tensor,)

    for proj, projection_tensor in zip(moe_spec.projections, projection_tensors):
        if config.is_mxfp8:
            q_weight, scale = batched_mx_cast_to_fp8(projection_tensor)
        else:
            q_weight, scale = batched_blockwise_cast_to_fp8(
                projection_tensor,
                config.weight_block_size,
                config.power_2_scale,
            )
        weight_name = f"{SKYRL_BATCHED_MOE_FP8_PREFIX}{moe_spec.experts_base}.{proj.hf_name}.weight"
        yield weight_name, q_weight
        yield scale_name_for_weight(weight_name, config.wire_format), scale


def iter_serialized_fp8_tensors(
    name: str,
    tensor: torch.Tensor,
    target_dtype: torch.dtype,
    config: SerializedFp8Config,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield vLLM checkpoint tensors for one Megatron-exported weight."""

    spec = config.require_spec()
    if spec.moe_expert_spec(name) is not None:
        yield from iter_batched_moe_expert_fp8_tensors(name, tensor, config)
        return

    if tensor.ndim == 2 and spec.should_quantize(name, tuple(tensor.shape), config.wire_format):
        if config.is_mxfp8:
            q_weight, scale = mx_cast_to_fp8(tensor)
        else:
            q_weight, scale = blockwise_cast_to_fp8(
                tensor,
                config.weight_block_size,
                config.power_2_scale,
            )
        yield name, q_weight
        yield scale_name_for_weight(name, config.wire_format), scale
        return

    yield name, tensor.to(dtype=target_dtype)
