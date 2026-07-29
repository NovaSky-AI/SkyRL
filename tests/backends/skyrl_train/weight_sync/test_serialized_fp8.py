from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.quantization import (
    SERIALIZED_BLOCKWISE_FP8,
    SERIALIZED_MXFP8,
    SERIALIZED_WEIGHT_PREFIX,
    BlockwiseFp8LinearStrategy,
    Mxfp8ExpertStrategy,
    batched_blockwise_cast_to_fp8,
    batched_mxfp8_cast_to_fp8,
    blockwise_cast_to_fp8,
    get_quantized_model_layout,
    get_qwen35_fp8_ignored_layers,
    get_serialized_fp8_quantization_config,
    get_serialized_mxfp8_quantization_config,
    iter_serialized_weight_tensors,
    mxfp8_cast_to_fp8,
)


def _wire_prefix(mode: str) -> str:
    return f"{SERIALIZED_WEIGHT_PREFIX}{mode}:"


def _serialize(name, tensor, *, mode, model_type, target_dtype=torch.bfloat16, strategy=None):
    strategy = strategy or (Mxfp8ExpertStrategy() if mode == SERIALIZED_MXFP8 else BlockwiseFp8LinearStrategy())
    layout = get_quantized_model_layout(model_type)
    return iter_serialized_weight_tensors(
        name,
        tensor,
        target_dtype,
        layout,
        strategy,
        model_type=model_type,
    )


def test_blockwise_cast_to_fp8_emits_weight_and_fp32_scale():
    weight = torch.arange(257 * 129, dtype=torch.float32).reshape(257, 129) / 1000
    q_weight, scale = blockwise_cast_to_fp8(weight, [128, 128])
    assert q_weight.shape == weight.shape
    assert q_weight.dtype == torch.float8_e4m3fn
    assert scale.shape == (3, 2)
    assert scale.dtype == torch.float32


def test_blockwise_cast_defaults_to_exact_fp32_scales():
    torch.manual_seed(7)
    weight = torch.randn(256, 256, dtype=torch.float32)
    default_weight, default_scale = blockwise_cast_to_fp8(weight, [128, 128])
    exact_weight, exact_scale = blockwise_cast_to_fp8(weight, [128, 128], power_2_scale=False)
    assert torch.equal(default_weight.view(torch.uint8), exact_weight.view(torch.uint8))
    assert torch.equal(default_scale, exact_scale)


def test_blockwise_cast_uses_training_amax_epsilon_for_near_zero_blocks(monkeypatch):
    monkeypatch.setenv("NVTE_FP8_BLOCK_AMAX_EPSILON", "1e-4")
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    strategy = BlockwiseFp8LinearStrategy()
    weight = torch.full((128, 128), 1e-8, dtype=torch.float32)
    _, scale = blockwise_cast_to_fp8(
        weight,
        strategy.weight_block_size,
        strategy.power_2_scale,
        strategy.amax_epsilon,
    )
    assert scale.item() == pytest.approx(strategy.amax_epsilon / torch.finfo(torch.float8_e4m3fn).max)


def test_blockwise_cast_pow2_scales_match_te_ue8m0_rule():
    torch.manual_seed(0)
    weight = torch.randn(256, 384, dtype=torch.float32)
    _, pow2_scale = blockwise_cast_to_fp8(weight, [128, 128], power_2_scale=True)
    _, exact_scale = blockwise_cast_to_fp8(weight, [128, 128], power_2_scale=False)
    expected = torch.pow(2.0, torch.ceil(torch.log2(exact_scale)))
    assert torch.allclose(torch.log2(pow2_scale), torch.log2(pow2_scale).round(), atol=0.0)
    assert torch.allclose(pow2_scale, expected)
    assert torch.all(pow2_scale >= exact_scale)


def test_blockwise_strategy_power_2_scale_follows_te_env(monkeypatch):
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    assert BlockwiseFp8LinearStrategy().power_2_scale is False
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    assert BlockwiseFp8LinearStrategy().power_2_scale is False
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0")
    assert BlockwiseFp8LinearStrategy().power_2_scale is True
    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "invalid")
    with pytest.raises(ValueError, match="must be '0'.*or '1'"):
        BlockwiseFp8LinearStrategy()


def test_blockwise_strategy_builds_runtime_env(monkeypatch):
    monkeypatch.delenv("VLLM_USE_DEEP_GEMM_E8M0", raising=False)
    strategy = BlockwiseFp8LinearStrategy(power_2_scale=False, amax_epsilon=1e-4)

    assert strategy.build_runtime_env() == {
        "NVTE_FP8_BLOCK_SCALING_FP32_SCALES": "1",
        "NVTE_FP8_BLOCK_AMAX_EPSILON": "0.0001",
        "VLLM_USE_DEEP_GEMM_E8M0": "0",
    }


@pytest.mark.parametrize("block_size", [[], [128], [128, 128, 128], [128, 0], [128, 1.5], [True, 128]])
def test_blockwise_cast_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="exactly two positive integers"):
        blockwise_cast_to_fp8(torch.ones((2, 2)), block_size)


def test_layout_classifies_quantizable_weights():
    layout = get_quantized_model_layout("qwen3_5_text")
    linear = torch.ones((256, 256), dtype=torch.bfloat16)
    embedding = torch.ones((32000, 256), dtype=torch.bfloat16)
    assert layout.weight_category("model.layers.0.mlp.down_proj.weight", linear.shape) == "linear"
    assert layout.weight_category("model.layers.0.linear_attn.in_proj_qkv.weight", linear.shape) == "linear"
    assert layout.weight_category("model.layers.0.linear_attn.conv1d.weight", linear.shape) is None
    assert layout.weight_category("model.layers.0.linear_attn.in_proj_b.weight", linear.shape) is None
    assert layout.weight_category("model.embed_tokens.weight", embedding.shape) is None

    tensors = list(
        _serialize(
            "model.embed_tokens.weight",
            embedding,
            mode=SERIALIZED_BLOCKWISE_FP8,
            model_type="qwen3_5_text",
        )
    )
    assert [(name, tensor.dtype) for name, tensor in tensors] == [("model.embed_tokens.weight", torch.bfloat16)]


def test_vllm_serialized_fp8_quantization_config():
    assert get_serialized_fp8_quantization_config() == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    assert get_serialized_fp8_quantization_config(ignored_layers=["model.layers.0.linear_attn.in_proj_b"])[
        "ignored_layers"
    ] == ["model.layers.0.linear_attn.in_proj_b"]


def test_vllm_serialized_mxfp8_quantization_config_is_expert_only():
    config = get_serialized_mxfp8_quantization_config()
    assert config["quant_method"] == "modelopt"
    assert config["quant_algo"] == "MXFP8"
    assert "*.self_attn.*" in config["ignore"]
    assert "*.mlp.shared_expert*" in config["ignore"]
    assert not any(pattern == "*.mlp.experts*" for pattern in config["ignore"])


def test_serialized_mxfp8_rejects_unknown_model_type():
    strategy = Mxfp8ExpertStrategy()
    with pytest.raises(ValueError, match="does not support model_type"):
        strategy.validate_model_type("llama")


def test_qwen35_fp8_ignored_layers_use_linear_attention_layers():
    hf_config = SimpleNamespace(
        model_type="qwen3_5_text",
        layer_types=["linear_attention", "full_attention", "linear_attention"],
    )
    assert get_qwen35_fp8_ignored_layers(hf_config) == [
        "model.layers.0.linear_attn.in_proj_b",
        "model.layers.0.linear_attn.in_proj_a",
        "model.language_model.layers.0.linear_attn.in_proj_b",
        "model.language_model.layers.0.linear_attn.in_proj_a",
        "model.layers.2.linear_attn.in_proj_b",
        "model.layers.2.linear_attn.in_proj_a",
        "model.language_model.layers.2.linear_attn.in_proj_b",
        "model.language_model.layers.2.linear_attn.in_proj_a",
    ]


def test_qwen35_ignored_layers_include_only_checkpoint_vision_prefixes():
    hf_config = SimpleNamespace(
        model_type="qwen3_5",
        text_config=SimpleNamespace(model_type="qwen3_5_text", layer_types=[]),
        vision_config=SimpleNamespace(depth=2),
    )
    assert get_qwen35_fp8_ignored_layers(hf_config) == [
        "model.visual.blocks.0.attn.proj",
        "model.visual.blocks.1.attn.proj",
    ]


def test_qwen35_ignored_layers_reject_unrelated_config():
    assert get_qwen35_fp8_ignored_layers(
        SimpleNamespace(model_type="unrelated_hybrid", layer_types=["linear_attention"])
    ) == []


def test_layout_splits_batched_gate_up_weight():
    layout = get_quantized_model_layout("qwen3_moe")
    base = "model.layers.5.mlp.experts"
    tensor = torch.randn(3, 256, 64)
    outputs = layout.split_exported_weight(f"{base}.gate_up_proj", tensor)
    assert [(name, value.shape) for name, value in outputs] == [
        (f"{base}.gate_proj.weight", torch.Size([3, 128, 64])),
        (f"{base}.up_proj.weight", torch.Size([3, 128, 64])),
    ]


def test_batched_blockwise_cast_matches_independent_expert_casts():
    torch.manual_seed(11)
    weight = torch.randn(5, 257, 129, dtype=torch.bfloat16)
    q_batched, scale_batched = batched_blockwise_cast_to_fp8(
        weight,
        [128, 128],
        power_2_scale=True,
        expert_batch_size=2,
    )
    for expert_id in range(weight.shape[0]):
        q_expected, scale_expected = blockwise_cast_to_fp8(
            weight[expert_id],
            [128, 128],
            power_2_scale=True,
        )
        assert torch.equal(q_batched[expert_id].view(torch.uint8), q_expected.view(torch.uint8))
        assert torch.equal(scale_batched[expert_id], scale_expected)


def test_mxfp8_cast_emits_row_major_e8m0_scales():
    weight = torch.cat(
        (
            torch.full((2, 32), 1.0, dtype=torch.bfloat16),
            torch.full((2, 32), 8.0, dtype=torch.bfloat16),
        ),
        dim=1,
    )
    q_weight, scale = mxfp8_cast_to_fp8(weight)
    expected_weight, expected_scale = blockwise_cast_to_fp8(weight, (1, 32), power_2_scale=True)
    assert q_weight.dtype == torch.float8_e4m3fn
    assert scale.dtype == torch.uint8
    assert torch.equal(scale, torch.tensor([[119, 122], [119, 122]], dtype=torch.uint8))
    assert torch.equal(q_weight.view(torch.uint8), expected_weight.view(torch.uint8))
    assert torch.equal(torch.exp2(scale.float() - 127), expected_scale)


def test_batched_mxfp8_cast_matches_batched_blockwise_cast():
    torch.manual_seed(17)
    weight = torch.randn(5, 96, 64, dtype=torch.bfloat16)
    q_batched, scale_batched = batched_mxfp8_cast_to_fp8(weight, expert_batch_size=2)
    q_expected, scale_expected = batched_blockwise_cast_to_fp8(
        weight,
        (1, 32),
        power_2_scale=True,
        expert_batch_size=2,
    )
    assert torch.equal(q_batched.view(torch.uint8), q_expected.view(torch.uint8))
    assert torch.equal(torch.exp2(scale_batched.float() - 127), scale_expected)


def test_batched_moe_experts_remain_fused_with_pow2_scales():
    num_experts, moe_inter, hidden = 3, 128, 256
    base = "model.language_model.layers.7.mlp.experts"
    strategy = BlockwiseFp8LinearStrategy(power_2_scale=True)
    torch.manual_seed(0)
    gate_up = torch.randn(num_experts, 2 * moe_inter, hidden, dtype=torch.bfloat16)
    emitted = dict(
        _serialize(
            f"{base}.gate_up_proj",
            gate_up,
            mode=SERIALIZED_BLOCKWISE_FP8,
            model_type="qwen3_5_moe_text",
            strategy=strategy,
        )
    )
    for projection_idx, projection in enumerate(("gate_proj", "up_proj")):
        weight_name = f"{_wire_prefix(SERIALIZED_BLOCKWISE_FP8)}{base}.{projection}.weight"
        weight = emitted[weight_name]
        scale = emitted[f"{weight_name}_scale_inv"]
        assert tuple(weight.shape) == (num_experts, moe_inter, hidden)
        assert tuple(scale.shape) == (num_experts, 1, 2)
        for expert_id in range(num_experts):
            source = gate_up[expert_id, projection_idx * moe_inter : (projection_idx + 1) * moe_inter]
            expected_weight, expected_scale = blockwise_cast_to_fp8(source, (128, 128), power_2_scale=True)
            assert torch.equal(weight[expert_id].view(torch.uint8), expected_weight.view(torch.uint8))
            assert torch.equal(scale[expert_id], expected_scale)


def test_qwen3_packed_expert_weights_serialize_as_mxfp8_only():
    base = "model.layers.2.mlp.experts"
    gate_up = torch.randn(3, 64, 64, dtype=torch.bfloat16)
    attention = torch.randn(128, 64, dtype=torch.bfloat16)
    experts = dict(
        _serialize(
            f"{base}.gate_up_proj",
            gate_up,
            mode=SERIALIZED_MXFP8,
            model_type="qwen3_moe",
        )
    )
    attention_tensors = list(
        _serialize(
            "model.layers.2.self_attn.q_proj.weight",
            attention,
            mode=SERIALIZED_MXFP8,
            model_type="qwen3_moe",
        )
    )
    assert set(experts) == {
        f"{_wire_prefix(SERIALIZED_MXFP8)}{base}.gate_proj.weight",
        f"{_wire_prefix(SERIALIZED_MXFP8)}{base}.gate_proj.weight_scale",
        f"{_wire_prefix(SERIALIZED_MXFP8)}{base}.up_proj.weight",
        f"{_wire_prefix(SERIALIZED_MXFP8)}{base}.up_proj.weight_scale",
    }
    assert [(name, tensor.dtype) for name, tensor in attention_tensors] == [
        ("model.layers.2.self_attn.q_proj.weight", torch.bfloat16)
    ]


def test_serialized_mxfp8_rejects_unregistered_expert_export_name():
    with pytest.raises(ValueError, match="Unsupported routed-expert export tensor"):
        list(
            _serialize(
                "model.layers.2.mlp.experts.unexpected.weight",
                torch.randn(128, 64),
                mode=SERIALIZED_MXFP8,
                model_type="qwen3_moe",
            )
        )


def test_batched_moe_gate_up_rejects_odd_output_dimension():
    with pytest.raises(ValueError, match="output dimension must be even"):
        list(
            _serialize(
                "model.layers.0.mlp.experts.gate_up_proj",
                torch.randn(2, 255, 128),
                mode=SERIALIZED_BLOCKWISE_FP8,
                model_type="qwen3_5_moe_text",
            )
        )


def test_batched_moe_expert_rejects_non_3d_input():
    with pytest.raises(ValueError, match="must be 3D"):
        list(
            _serialize(
                "model.layers.0.mlp.experts.down_proj",
                torch.ones((128, 128)),
                mode=SERIALIZED_BLOCKWISE_FP8,
                model_type="qwen3_5_moe_text",
            )
        )
