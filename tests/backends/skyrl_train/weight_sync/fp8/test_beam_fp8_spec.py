"""Beam ``ModelFp8Spec``: quantize / BF16 split, vLLM ignore matching, and parity with the released FP8 checkpoint.

Run with:
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/weight_sync/fp8/test_beam_fp8_spec.py
"""

import json
import os
from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.weight_sync.fp8 import resolve_fp8_spec
from skyrl.backends.skyrl_train.weight_sync.fp8.models import BEAM_FP8_SPEC
from skyrl.backends.skyrl_train.weight_sync.fp8.models.base import BLOCKWISE_FP8, MXFP8

# Optional local copies of the two checkpoints' weight indexes (no weights needed).
BF16_INDEX = os.environ.get("BEAM_BF16_INDEX", "")
FP8_INDEX = os.environ.get("BEAM_FP8_INDEX", "")


def test_beam_spec_resolves_for_beam_only():
    assert resolve_fp8_spec(SimpleNamespace(model_type="beam")) is BEAM_FP8_SPEC
    assert resolve_fp8_spec(SimpleNamespace(model_type="glm_moe_dsa")) is not BEAM_FP8_SPEC


@pytest.mark.parametrize("wire", [BLOCKWISE_FP8, MXFP8])
@pytest.mark.parametrize(
    "name,shape",
    [
        ("model.layers.0.self_attn.q_proj.weight", (6144, 5120)),
        ("model.layers.0.self_attn.k_proj.weight", (1024, 5120)),
        ("model.layers.0.self_attn.v_proj.weight", (1024, 5120)),
        ("model.layers.0.self_attn.gate_proj.weight", (6144, 5120)),
        ("model.layers.0.self_attn.o_proj.weight", (5120, 6144)),
        ("model.layers.0.mlp.gate_proj.weight", (20480, 5120)),
        ("model.layers.0.mlp.up_proj.weight", (20480, 5120)),
        ("model.layers.0.mlp.down_proj.weight", (5120, 20480)),
        ("model.layers.7.mlp.experts.255.gate_proj.weight", (2560, 5120)),
        ("model.layers.7.mlp.experts.0.up_proj.weight", (2560, 5120)),
        ("model.layers.7.mlp.experts.3.down_proj.weight", (5120, 2560)),
    ],
)
def test_beam_linears_serialize_as_fp8(name, shape, wire):
    assert BEAM_FP8_SPEC.should_quantize(name, shape, wire)


@pytest.mark.parametrize(
    "name,shape",
    [
        ("model.layers.7.mlp.gate.weight", (256, 5120)),  # fp32 router
        ("model.layers.7.mlp.gate.e_score_correction_bias", (256,)),
        ("model.layers.0.self_attn.gate_bias", (6144,)),
        ("model.layers.0.self_attn.q_norm.weight", (128,)),
        ("model.layers.0.input_layernorm.weight", (5120,)),
        ("model.embed_tokens.weight", (151936, 5120)),
        ("lm_head.weight", (151936, 5120)),
        ("model.norm.weight", (5120,)),
    ],
)
def test_beam_bf16_weights_stay_unquantized(name, shape):
    assert not BEAM_FP8_SPEC.should_quantize(name, shape, BLOCKWISE_FP8)
    assert not BEAM_FP8_SPEC.should_quantize(name, shape, MXFP8)


def test_beam_mxfp8_requires_32_aligned_reduction_dim():
    assert not BEAM_FP8_SPEC.should_quantize("model.layers.0.self_attn.o_proj.weight", (5120, 6144 + 16), MXFP8)
    assert BEAM_FP8_SPEC.should_quantize("model.layers.0.self_attn.o_proj.weight", (5120, 6144 + 16), BLOCKWISE_FP8)


def test_beam_spec_has_no_batched_experts():
    assert BEAM_FP8_SPEC.moe_expert_spec("model.layers.7.mlp.experts.gate_up_proj") is None
    assert BEAM_FP8_SPEC.moe_projections == ()


def test_beam_ignore_matches_vllm_modules():
    """vLLM's compressed-tensors matcher must skip the router and leave every attention / MLP linear (incl. fused)."""
    pytest.importorskip("vllm")
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    for wire in (BLOCKWISE_FP8, MXFP8):
        ignore = BEAM_FP8_SPEC.ignored_layers(SimpleNamespace(model_type="beam"), wire)
        # vLLM's Beam packed modules (BeamForCausalLM.packed_modules_mapping).
        fused = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "qkv_gate_proj": ["q_proj", "k_proj", "v_proj", "output_gate_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        assert should_ignore_layer("model.layers.7.mlp.gate", ignore, fused)
        for name in (
            "model.layers.0.self_attn.qkv_gate_proj",
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.self_attn.output_gate_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_up_proj",
            "model.layers.0.mlp.down_proj",
        ):
            assert not should_ignore_layer(name, ignore, fused), name


@pytest.mark.skipif(
    not (os.path.exists(BF16_INDEX) and os.path.exists(FP8_INDEX)), reason="checkpoint indexes not present"
)
def test_beam_spec_matches_released_fp8_checkpoint():
    """Every BF16-checkpoint weight the spec quantizes is FP8 (has a weight_scale) in the released FP8 checkpoint,
    and vice versa. All Beam dims are 32-aligned, so a placeholder 2D shape exercises the name rules."""
    bf16_names = json.load(open(BF16_INDEX))["weight_map"]
    fp8_map = json.load(open(FP8_INDEX))["weight_map"]
    fp8_quantized = {n[: -len("_scale")] for n in fp8_map if n.endswith(".weight_scale")}

    def two_d(n):
        return (128, 5120) if n.endswith(".weight") and not n.endswith(("norm.weight",)) else (5120,)

    ours = {n for n in bf16_names if BEAM_FP8_SPEC.should_quantize(n, two_d(n), MXFP8)}
    assert ours == fp8_quantized, (sorted(ours - fp8_quantized)[:5], sorted(fp8_quantized - ours)[:5])
    assert len(ours) == 52 * 5 + 3 * 3 + 49 * 256 * 3
