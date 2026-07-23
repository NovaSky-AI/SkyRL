from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.serialized_fp8 import (
    SKYRL_BATCHED_MOE_FP8_PREFIX,
    SerializedFp8Config,
    batched_blockwise_cast_to_fp8,
    batched_moe_expert_spec,
    blockwise_cast_to_fp8,
    get_qwen35_fp8_ignored_layers,
    get_serialized_fp8_quantization_config,
    is_quantizable_weight,
    is_quantizable_weight_shape,
    iter_serialized_fp8_tensors,
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
    config = SerializedFp8Config()
    weight = torch.full((128, 128), 1e-8, dtype=torch.float32)

    _, scale = blockwise_cast_to_fp8(
        weight,
        config.weight_block_size,
        config.power_2_scale,
        config.amax_epsilon,
    )

    assert scale.item() == pytest.approx(config.amax_epsilon / torch.finfo(torch.float8_e4m3fn).max)


def test_blockwise_cast_pow2_scales_match_te_ue8m0_rule():
    # TE's UE8M0 mode rounds dequantization scales up to powers of two.
    torch.manual_seed(0)
    weight = torch.randn(256, 384, dtype=torch.float32)

    _, pow2_scale = blockwise_cast_to_fp8(weight, [128, 128], power_2_scale=True)
    _, exact_scale = blockwise_cast_to_fp8(weight, [128, 128], power_2_scale=False)

    log2 = torch.log2(pow2_scale)
    assert torch.allclose(log2, log2.round(), atol=0.0)
    expected = torch.pow(2.0, torch.ceil(torch.log2(exact_scale)))
    assert torch.allclose(pow2_scale, expected)
    assert torch.all(pow2_scale >= exact_scale)


def test_serialized_config_power_2_scale_follows_te_env(monkeypatch):
    monkeypatch.delenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", raising=False)
    assert SerializedFp8Config().power_2_scale is False

    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "1")
    assert SerializedFp8Config().power_2_scale is False

    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "0")
    assert SerializedFp8Config().power_2_scale is True

    monkeypatch.setenv("NVTE_FP8_BLOCK_SCALING_FP32_SCALES", "invalid")
    with pytest.raises(ValueError, match="must be '0'.*or '1'"):
        SerializedFp8Config()


@pytest.mark.parametrize("block_size", [[], [128], [128, 128, 128], [128, 0], [128, 1.5], [True, 128]])
def test_blockwise_cast_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="exactly two positive integers"):
        blockwise_cast_to_fp8(torch.ones((2, 2)), block_size)


def test_quantizable_weight_filter_keeps_embeddings_in_target_dtype():
    config = SerializedFp8Config()
    linear = torch.ones((256, 256), dtype=torch.bfloat16)
    embedding = torch.ones((32000, 256), dtype=torch.bfloat16)

    assert is_quantizable_weight("model.layers.0.mlp.down_proj.weight", linear)
    assert is_quantizable_weight("model.layers.0.linear_attn.in_proj_qkv.weight", linear)
    assert is_quantizable_weight("model.layers.0.linear_attn.in_proj_z.weight", linear)
    assert is_quantizable_weight("model.layers.0.linear_attn.out_proj.weight", linear)
    assert not is_quantizable_weight("model.layers.0.linear_attn.conv1d.weight", linear)
    assert not is_quantizable_weight("model.layers.0.linear_attn.in_proj_b.weight", linear)
    assert not is_quantizable_weight("model.layers.0.linear_attn.in_proj_a.weight", linear)
    assert not is_quantizable_weight("model.embed_tokens.weight", embedding)

    tensors = list(
        iter_serialized_fp8_tensors(
            "model.embed_tokens.weight",
            embedding,
            torch.bfloat16,
            config,
        )
    )
    assert [(name, tensor.dtype) for name, tensor in tensors] == [("model.embed_tokens.weight", torch.bfloat16)]


def test_vllm_serialized_fp8_quantization_config():
    assert get_serialized_fp8_quantization_config() == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
    }
    assert get_serialized_fp8_quantization_config(ignored_layers=["model.layers.0.linear_attn.in_proj_b"]) == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "ignored_layers": ["model.layers.0.linear_attn.in_proj_b"],
    }


def test_qwen35_fp8_ignored_layers_use_linear_attention_layers():
    hf_config = SimpleNamespace(
        model_type="qwen3_5_text",
        layer_types=[
            "linear_attention",
            "full_attention",
            "linear_attention",
        ],
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


def test_qwen35_ignored_layers_are_not_inferred_from_unrelated_hybrid_config():
    hf_config = SimpleNamespace(model_type="unrelated_hybrid", layer_types=["linear_attention"])

    assert get_qwen35_fp8_ignored_layers(hf_config) == []


def test_moe_batched_expert_spec_recognizes_and_splits_gate_up():
    base = "model.language_model.layers.5.mlp.experts"
    assert batched_moe_expert_spec(f"{base}.gate_up_proj") == (base, ("gate_proj", "up_proj"), True)
    assert batched_moe_expert_spec(f"{base}.down_proj") == (base, ("down_proj",), False)
    assert batched_moe_expert_spec("model.layers.5.mlp.gate.weight") is None
    assert batched_moe_expert_spec("model.layers.5.self_attn.q_proj.weight") is None


def test_moe_shared_expert_quantized_router_and_gate_bf16():
    lin = (256, 256)
    assert is_quantizable_weight_shape("model.layers.3.mlp.shared_expert.gate_proj.weight", lin)
    assert is_quantizable_weight_shape("model.layers.3.mlp.shared_expert.up_proj.weight", lin)
    assert is_quantizable_weight_shape("model.layers.3.mlp.shared_expert.down_proj.weight", lin)
    assert not is_quantizable_weight_shape("model.layers.3.mlp.gate.weight", (256, 8))
    assert not is_quantizable_weight_shape("model.layers.3.mlp.shared_expert_gate.weight", (256, 1))


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


def test_batched_moe_experts_remain_fused_with_pow2_scales():
    num_experts, moe_inter, hidden = 3, 128, 256
    config = SerializedFp8Config(weight_block_size=(128, 128), power_2_scale=True)
    base = "model.language_model.layers.7.mlp.experts"

    torch.manual_seed(0)
    gate_up = torch.randn(num_experts, 2 * moe_inter, hidden, dtype=torch.bfloat16)
    emitted = list(iter_serialized_fp8_tensors(f"{base}.gate_up_proj", gate_up, torch.bfloat16, config))
    by_name = dict(emitted)
    assert len(emitted) == 4
    for projection_idx, proj in enumerate(("gate_proj", "up_proj")):
        weight_name = f"{SKYRL_BATCHED_MOE_FP8_PREFIX}{base}.{proj}.weight"
        w = by_name[weight_name]
        s = by_name[f"{SKYRL_BATCHED_MOE_FP8_PREFIX}{base}.{proj}.weight_scale_inv"]
        assert w.dtype == torch.float8_e4m3fn and tuple(w.shape) == (num_experts, moe_inter, hidden)
        assert s.dtype == torch.float32 and tuple(s.shape) == (num_experts, 1, 2)
        log2 = torch.log2(s)
        assert torch.allclose(log2, log2.round(), atol=0.0)
        for expert_id in range(num_experts):
            source = gate_up[expert_id, projection_idx * moe_inter : (projection_idx + 1) * moe_inter]
            q_expected, scale_expected = blockwise_cast_to_fp8(
                source,
                config.weight_block_size,
                config.power_2_scale,
            )
            assert torch.equal(w[expert_id].view(torch.uint8), q_expected.view(torch.uint8))
            assert torch.equal(s[expert_id], scale_expected)

    down = torch.randn(num_experts, hidden, moe_inter, dtype=torch.bfloat16)
    emitted_d = dict(iter_serialized_fp8_tensors(f"{base}.down_proj", down, torch.bfloat16, config))
    down_weight_name = f"{SKYRL_BATCHED_MOE_FP8_PREFIX}{base}.down_proj.weight"
    assert set(emitted_d) == {
        down_weight_name,
        f"{SKYRL_BATCHED_MOE_FP8_PREFIX}{base}.down_proj.weight_scale_inv",
    }
    assert emitted_d[down_weight_name].shape == down.shape


def test_batched_moe_gate_up_rejects_odd_output_dimension():
    name = "model.layers.0.mlp.experts.gate_up_proj"
    tensor = torch.randn(2, 255, 128, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="output dimension must be even"):
        list(iter_serialized_fp8_tensors(name, tensor, torch.bfloat16, SerializedFp8Config()))


def test_batched_moe_expert_iterator_rejects_non_3d_input():
    name = "model.layers.0.mlp.experts.down_proj"
    with pytest.raises(ValueError, match="must be 3D"):
        list(
            iter_serialized_fp8_tensors(
                name,
                torch.ones((128, 128)),
                torch.bfloat16,
                SerializedFp8Config(),
            )
        )
