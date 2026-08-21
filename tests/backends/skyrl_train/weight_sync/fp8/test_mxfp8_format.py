"""MXFP8 wire format: quantizer semantics, format selection, and the vLLM contract.

The bitwise-identity check against Transformer Engine lives in
``tests/backends/skyrl_train/gpu/gpu_ci`` because it needs a Blackwell device;
everything here is CPU-only.
"""

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.fp8 import (
    BLOCKWISE_FP8,
    MXFP8,
    MXFP8_GROUP_SIZE,
    SerializedFp8Config,
    batched_moe_wire_targets,
    batched_mx_cast_to_fp8,
    get_serialized_fp8_quantization_config,
    mx_cast_to_fp8,
    scale_name_for_weight,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.models.qwen35 import (
    get_qwen35_fp8_ignored_layers,
    is_quantizable_weight_shape,
)

E4M3_MAX = 448.0


def _dequantize(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rows, cols = codes.shape
    grouped = codes.to(torch.float32).view(rows, cols // MXFP8_GROUP_SIZE, MXFP8_GROUP_SIZE)
    factor = torch.pow(2.0, scales.to(torch.float32) - 127.0)
    return (grouped * factor[..., None]).view(rows, cols)


def test_mx_cast_shapes_and_dtypes():
    codes, scales = mx_cast_to_fp8(torch.randn(64, 256))
    assert codes.shape == (64, 256)
    assert codes.dtype == torch.float8_e4m3fn
    assert scales.shape == (64, 256 // MXFP8_GROUP_SIZE)
    assert scales.dtype == torch.uint8


def test_mx_cast_uses_ceil_log2_amax_over_448():
    """The shared exponent is TE's rule; a deliberate amax pins the expected byte."""

    weight = torch.zeros(1, MXFP8_GROUP_SIZE)
    weight[0, 0] = 448.0  # amax/448 == 1 -> exponent 0 -> biased 127
    _, scales = mx_cast_to_fp8(weight)
    assert scales[0, 0].item() == 127

    weight[0, 0] = 449.0  # just above -> ceil pushes the exponent to 1
    _, scales = mx_cast_to_fp8(weight)
    assert scales[0, 0].item() == 128


def test_mx_cast_roundtrip_is_close():
    weight = torch.randn(32, 128)
    codes, scales = mx_cast_to_fp8(weight)
    recovered = _dequantize(codes, scales)
    assert torch.allclose(recovered, weight, rtol=0.15, atol=1e-2)


def test_mx_cast_zero_group_is_finite():
    """A zero-token MoE expert produces all-zero blocks; the scale must stay representable."""

    weight = torch.zeros(4, MXFP8_GROUP_SIZE * 2)
    codes, scales = mx_cast_to_fp8(weight)
    assert torch.isfinite(_dequantize(codes, scales)).all()
    assert (_dequantize(codes, scales) == 0).all()


def test_mx_cast_rejects_unaligned_reduction_dim():
    with pytest.raises(ValueError, match="multiple of 32"):
        mx_cast_to_fp8(torch.randn(8, 48))


def test_batched_mx_cast_matches_per_expert():
    experts = torch.randn(5, 32, 64)
    batched_codes, batched_scales = batched_mx_cast_to_fp8(experts, expert_batch_size=2)
    for i in range(experts.shape[0]):
        codes, scales = mx_cast_to_fp8(experts[i])
        assert torch.equal(batched_codes[i].view(torch.uint8), codes.view(torch.uint8))
        assert torch.equal(batched_scales[i], scales)


def test_quantization_config_matches_vllm_is_mxfp8_predicate():
    """vLLM selects both its dense and fused-MoE MXFP8 schemes from this exact predicate."""

    config = get_serialized_fp8_quantization_config(ignored_layers=["lm_head"], wire_format=MXFP8)
    assert config["quant_method"] == "compressed-tensors"
    assert config["ignore"] == ["lm_head"]
    weights = config["config_groups"]["group_0"]["weights"]
    assert weights["strategy"] == "group"
    assert weights["group_size"] == 32
    assert weights["symmetric"] is True
    assert weights["type"] == "float"
    assert weights["num_bits"] == 8
    assert weights["scale_dtype"] == "uint8"
    assert config["config_groups"]["group_0"]["input_activations"]["dynamic"] is True


def test_blockwise_quantization_config_is_unchanged():
    """The shipped blockwise path must not shift; its runs are already validated."""

    assert get_serialized_fp8_quantization_config(ignored_layers=["a"]) == {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "ignored_layers": ["a"],
    }


def test_scale_names_differ_by_format():
    assert scale_name_for_weight("m.q_proj.weight") == "m.q_proj.weight_scale_inv"
    assert scale_name_for_weight("m.q_proj.weight", MXFP8) == "m.q_proj.weight_scale"


def test_config_rejects_unknown_wire_format():
    assert SerializedFp8Config(wire_format=MXFP8).is_mxfp8 is True
    assert SerializedFp8Config().is_mxfp8 is False
    with pytest.raises(ValueError, match="wire_format must be one of"):
        SerializedFp8Config(wire_format="int4")


def test_should_quantize_requires_group_aligned_reduction_dim_for_mxfp8():
    name = "model.layers.0.self_attn.q_proj.weight"
    assert is_quantizable_weight_shape(name, (4096, 4096), BLOCKWISE_FP8) is True
    assert is_quantizable_weight_shape(name, (4096, 4096), MXFP8) is True
    # The vision hidden size leaves a remainder of 16 under MXFP8's 32-wide groups.
    assert is_quantizable_weight_shape(name, (4096, 4304), MXFP8) is False


def test_mxfp8_ignores_extend_blockwise_without_changing_it():
    hf_config = type(
        "Cfg",
        (),
        {
            "model_type": "qwen3_5_moe",
            "layer_types": ["linear_attention", "full_attention"],
            "vision_config": type("V", (), {"depth": 2})(),
        },
    )()

    blockwise = get_qwen35_fp8_ignored_layers(hf_config, BLOCKWISE_FP8)
    mxfp8 = get_qwen35_fp8_ignored_layers(hf_config, MXFP8)

    assert set(blockwise).issubset(set(mxfp8))
    extra = set(mxfp8) - set(blockwise)
    assert extra == {
        "model.visual.blocks.0.attn.qkv",
        "model.visual.blocks.1.attn.qkv",
        "model.visual.merger.linear_fc1",
        "model.visual.merger.linear_fc2",
    }
    # GDN small-N projections are excluded under both formats.
    for fmt_list in (blockwise, mxfp8):
        assert "model.layers.0.linear_attn.in_proj_a" in fmt_list
        assert "model.layers.0.linear_attn.in_proj_b" in fmt_list


def test_moe_wire_targets_cover_both_scale_suffixes():
    targets = batched_moe_wire_targets()
    assert targets[".experts.gate_proj.weight_scale_inv"] == (".experts.w13_weight_scale_inv", "w1")
    assert targets[".experts.gate_proj.weight_scale"] == (".experts.w13_weight_scale", "w1")
    assert targets[".experts.down_proj.weight_scale"] == (".experts.w2_weight_scale", "w2")
