from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import torch

from skyrl.backends.skyrl_train.quantization import (
    BLOCKWISE_FP8,
    MXFP8,
    QWEN35_LAYOUT,
    BlockwiseFp8Strategy,
    ModelQuantizationLayout,
    Mxfp8ExpertStrategy,
    QuantizationStrategy,
    QuantizationTarget,
    blockwise_cast_to_fp8,
    build_mxfp8_te_recipe,
    decode_serialized_name,
    get_quantized_model_layout,
    get_serialized_weight_strategy,
    iter_serialized_weight_tensors,
    normalize_block_size,
)
from skyrl.backends.skyrl_train.quantization.vllm import (
    resolve_registered_vllm_receiver_target,
    resolve_vllm_receiver_target,
)


class _CustomStrategy(QuantizationStrategy):
    mode = "custom"
    target_names = frozenset({"arbitrary_projection"})
    receiver_suffixes: ClassVar[dict[str, str]] = {".weight": ""}
    vllm_quantization = "custom"

    def serialize_weight(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        batched_experts: bool,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        del batched_experts
        yield f"{name}.quantized", tensor.to(torch.float16)

    def vllm_quantization_config(
        self,
        inference_config: Any,
        hf_config: Any,
        layout: ModelQuantizationLayout,
    ) -> dict[str, Any]:
        del inference_config, hf_config, layout
        return {}


def test_model_layout_supports_arbitrary_target_modules():
    layout = ModelQuantizationLayout(
        name="custom",
        model_types=frozenset({"custom"}),
        targets=(
            QuantizationTarget(
                name="arbitrary_projection",
                matches_exported_weight=lambda name, shape: (name.endswith(".special.weight") and len(shape) == 2),
                megatron_patterns=("*.special",),
            ),
        ),
    )
    emitted = list(
        iter_serialized_weight_tensors(
            "model.special.weight",
            torch.ones((4, 4), dtype=torch.bfloat16),
            torch.bfloat16,
            layout,
            _CustomStrategy(),
            model_type="custom",
        )
    )
    assert [(name, tensor.dtype) for name, tensor in emitted] == [("model.special.weight.quantized", torch.float16)]


def test_blockwise_mode_resolves_strategy():
    assert isinstance(get_serialized_weight_strategy(BLOCKWISE_FP8), BlockwiseFp8Strategy)


def test_blockwise_cast_handles_padding_and_zero_blocks():
    weight = torch.zeros((129, 257), dtype=torch.float32)
    q_weight, scale = blockwise_cast_to_fp8(weight, (128, 128))
    assert q_weight.shape == weight.shape
    assert q_weight.dtype == torch.float8_e4m3fn
    assert scale.shape == (2, 3)
    assert torch.isfinite(scale).all()


def test_blockwise_power_of_two_scales():
    torch.manual_seed(0)
    weight = torch.randn((256, 384), dtype=torch.float32)
    _, scales = blockwise_cast_to_fp8(weight, (128, 128), power_2_scale=True)
    assert torch.equal(torch.log2(scales), torch.log2(scales).round())


@pytest.mark.parametrize("block_size", [(), (128,), (128, 0), (128, 1.5), (True, 128)])
def test_blockwise_rejects_invalid_block_sizes(block_size):
    with pytest.raises(ValueError, match="exactly two positive integers"):
        normalize_block_size(block_size)


def test_generic_blockwise_strategy_emits_expert_weight_and_scale():
    source = torch.randn((2, 64, 128), dtype=torch.bfloat16)
    name = "model.layers.0.mlp.experts.down_proj"
    generic = list(
        iter_serialized_weight_tensors(
            name,
            source,
            torch.bfloat16,
            QWEN35_LAYOUT,
            BlockwiseFp8Strategy(power_2_scale=True),
            model_type="qwen3_5_moe_text",
        )
    )
    assert [decode_serialized_name(generic_name)[1] for generic_name, _tensor in generic] == [
        "model.layers.0.mlp.experts.down_proj.weight",
        "model.layers.0.mlp.experts.down_proj.weight_scale_inv",
    ]
    assert generic[0][1].dtype == torch.float8_e4m3fn
    assert generic[1][1].dtype == torch.float32


def test_qwen35_layout_classifies_dense_and_expert_targets():
    layout = get_quantized_model_layout(SimpleNamespace(model_type="qwen3_5_moe_text"))
    assert layout is QWEN35_LAYOUT
    assert layout.weight_target("model.layers.0.self_attn.q_proj.weight", (128, 128)) == "linear"
    assert layout.weight_target("model.layers.0.mlp.experts.gate_proj.weight", (4, 128, 128)) == ("routed_expert_gate")
    assert layout.weight_target("model.embed_tokens.weight", (32000, 128)) is None


def test_persistent_mxfp8_recipe_only_changes_parameter_storage():
    transient = build_mxfp8_te_recipe(persistent=False)
    persistent = build_mxfp8_te_recipe(persistent=True)
    assert transient["training_recipe"]["fp8_param"] is False
    assert transient["evaluation_recipe"]["fp8_param"] is False
    assert persistent["training_recipe"]["fp8_param"] is True
    assert persistent["evaluation_recipe"]["fp8_param"] is True


def test_mxfp8_exports_e8m0_runtime_contract():
    assert Mxfp8ExpertStrategy().build_runtime_env() == {
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "0",
        "VLLM_USE_DEEP_GEMM_E8M0": "1",
    }


def test_mxfp8_targets_only_routed_experts():
    strategy = Mxfp8ExpertStrategy()
    dense = list(
        iter_serialized_weight_tensors(
            "model.layers.0.self_attn.q_proj.weight",
            torch.ones((32, 32), dtype=torch.bfloat16),
            torch.bfloat16,
            QWEN35_LAYOUT,
            strategy,
            model_type="qwen3_5_moe_text",
        )
    )
    assert [(name, tensor.dtype) for name, tensor in dense] == [
        ("model.layers.0.self_attn.q_proj.weight", torch.bfloat16)
    ]

    source = torch.ones((2, 64, 32), dtype=torch.bfloat16)
    emitted = list(
        iter_serialized_weight_tensors(
            "model.layers.0.mlp.experts.down_proj",
            source,
            torch.bfloat16,
            QWEN35_LAYOUT,
            strategy,
            model_type="qwen3_5_moe_text",
        )
    )
    decoded = [decode_serialized_name(name) for name, _tensor in emitted]
    assert [mode for mode, _name in decoded] == [MXFP8, MXFP8]
    assert emitted[0][1].dtype == torch.float8_e4m3fn
    assert emitted[1][1].dtype == torch.uint8


def test_layout_and_strategy_resolve_vllm_expert_target():
    strategy = Mxfp8ExpertStrategy()
    assert resolve_vllm_receiver_target(
        strategy,
        QWEN35_LAYOUT,
        "model.layers.0.mlp.experts.up_proj.weight_scale",
    ) == ("model.layers.0.mlp.experts.w13_weight_scale", "w3")
    assert resolve_registered_vllm_receiver_target(
        MXFP8,
        "model.layers.0.mlp.experts.up_proj.weight_scale",
    ) == ("model.layers.0.mlp.experts.w13_weight_scale", "w3")
