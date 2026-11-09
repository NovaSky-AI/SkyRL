"""Beam ``ModelFp8Spec``: the quantize / unquantized split, the FP8 module list, and parity with the
released FP8 checkpoint (``reflection/Beam-1.0-501B-A23B-FP8``).

Run with:
uv run --isolated --extra dev --extra skyrl-train pytest tests/backends/skyrl_train/weight_sync/fp8/test_beam_fp8_spec.py
"""

import json
import os
from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.weight_sync.fp8 import (
    BLOCKWISE_FP8,
    MXFP8,
    resolve_fp8_spec,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.models import BEAM_FP8_SPEC

# Optional local copy of the released FP8 checkpoint's model.safetensors.index.json (no weights needed).
FP8_INDEX = os.environ.get("BEAM_FP8_INDEX", "")


def _beam_config(num_experts=4, mlp_layer_types=("dense", "sparse", "sparse")):
    return SimpleNamespace(model_type="beam", num_experts=num_experts, mlp_layer_types=list(mlp_layer_types))


def test_beam_spec_resolves_for_beam_only():
    assert resolve_fp8_spec(SimpleNamespace(model_type="beam")) is BEAM_FP8_SPEC
    assert not BEAM_FP8_SPEC.matches(SimpleNamespace(model_type="glm_moe_dsa"))


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
        ("model.layers.0.self_attn.gate_bias", (48,)),
        ("model.layers.0.self_attn.q_norm.weight", (128,)),
        ("model.layers.0.input_layernorm.weight", (5120,)),
        ("model.layers.0.post_feedforward_layernorm.weight", (5120,)),
        ("model.embed_tokens.weight", (151936, 5120)),
        ("lm_head.weight", (151936, 5120)),
        ("model.norm.weight", (5120,)),
    ],
)
def test_beam_other_weights_stay_unquantized(name, shape):
    assert not BEAM_FP8_SPEC.should_quantize(name, shape, BLOCKWISE_FP8)
    assert not BEAM_FP8_SPEC.should_quantize(name, shape, MXFP8)


def test_beam_mxfp8_requires_32_aligned_reduction_dim():
    name = "model.layers.0.self_attn.o_proj.weight"
    assert not BEAM_FP8_SPEC.should_quantize(name, (5120, 6144 + 16), MXFP8)
    assert BEAM_FP8_SPEC.should_quantize(name, (5120, 6144 + 16), BLOCKWISE_FP8)


@pytest.mark.parametrize("wire", [BLOCKWISE_FP8, MXFP8])
def test_beam_fp8_modules_follow_mlp_layer_types(wire):
    modules = BEAM_FP8_SPEC.fp8_modules(_beam_config(), wire)
    attention = ["q_proj", "k_proj", "v_proj", "gate_proj", "o_proj"]
    expected = [f"model.layers.0.self_attn.{m}" for m in attention]
    expected += [f"model.layers.0.mlp.{m}" for m in ("gate_proj", "up_proj", "down_proj")]
    for layer in (1, 2):
        expected += [f"model.layers.{layer}.self_attn.{m}" for m in attention]
        expected += [
            f"model.layers.{layer}.mlp.experts.{e}.{p}" for e in range(4) for p in ("gate_proj", "up_proj", "down_proj")
        ]
    assert modules == expected
    # Every listed module is one the sender quantizes.
    assert all(BEAM_FP8_SPEC.should_quantize(f"{m}.weight", (128, 256), wire) for m in modules)


def test_beam_spec_has_no_stacked_experts():
    assert BEAM_FP8_SPEC.stacked_experts("model.layers.7.mlp.experts.0.gate_proj.weight") is None


def test_beam_base_exclude_list_mirrors_released_checkpoint():
    for wire in (BLOCKWISE_FP8, MXFP8):
        assert BEAM_FP8_SPEC.base_exclude_list(_beam_config(), wire) == [
            "lm_head",
            "model.embed_tokens",
            "re:.*mlp\\.gate$",
        ]


@pytest.mark.skipif(not os.path.exists(FP8_INDEX), reason="BEAM_FP8_INDEX not set")
def test_beam_fp8_modules_match_released_fp8_checkpoint():
    """The FP8 modules for the released config are exactly the tensors the FP8 checkpoint
    quantizes (those with a ``weight_scale``)."""
    fp8_map = json.load(open(FP8_INDEX))["weight_map"]
    released = {n[: -len(".weight_scale")] for n in fp8_map if n.endswith(".weight_scale")}
    config = _beam_config(num_experts=256, mlp_layer_types=["dense"] * 3 + ["sparse"] * 49)
    assert set(BEAM_FP8_SPEC.fp8_modules(config, MXFP8)) == released
