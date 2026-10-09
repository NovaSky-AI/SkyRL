"""User FP8 exclusions (``fp8_weight_sync_exclude_modules``) on top of the Qwen3.5 spec.

One expansion (``resolve_user_provided_exclude_list``) feeds both the engine's exclude
list and the sender. These tests cover the expansion and each of its two consumers.
"""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.fp8 import (
    BLOCKWISE_FP8,
    MXFP8,
    SKYRL_BATCHED_MOE_FP8_PREFIX,
    SerializedFp8Config,
    engine_exclude_list,
    iter_serialized_fp8_tensors,
    resolve_serialized_fp8_config,
    resolve_user_provided_exclude_list,
)
from skyrl.backends.skyrl_train.weight_sync.fp8.models import QWEN35_FP8_SPEC

LM = "model.language_model.layers"


def _moe_vl_config():
    """A unified VL Qwen3.5 MoE config: three GDN layers, then one full-attention layer."""
    return SimpleNamespace(
        model_type="qwen3_5_moe",
        text_config=SimpleNamespace(
            model_type="qwen3_5_moe_text",
            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            shared_expert_intermediate_size=512,
        ),
    )


def _config(*excluded, wire_format=BLOCKWISE_FP8):
    return SerializedFp8Config(
        spec=QWEN35_FP8_SPEC, wire_format=wire_format, user_provided_exclude_list=frozenset(excluded)
    )


def test_fp8_modules_use_the_bridge_export_names():
    modules = QWEN35_FP8_SPEC.fp8_modules(_moe_vl_config())

    assert modules[:7] == [
        f"{LM}.0.linear_attn.in_proj_qkv",
        f"{LM}.0.linear_attn.in_proj_z",
        f"{LM}.0.linear_attn.out_proj",
        f"{LM}.0.mlp.experts",
        f"{LM}.0.mlp.shared_expert.gate_proj",
        f"{LM}.0.mlp.shared_expert.up_proj",
        f"{LM}.0.mlp.shared_expert.down_proj",
    ]
    assert modules[21:25] == [f"{LM}.3.self_attn.{proj}" for proj in ("q_proj", "k_proj", "v_proj", "o_proj")]
    assert len(modules) == 3 * 7 + 8


def test_fp8_modules_of_a_flat_dense_text_config():
    hf_config = SimpleNamespace(model_type="qwen3_5_text", layer_types=["full_attention"])

    assert QWEN35_FP8_SPEC.fp8_modules(hf_config) == [
        "model.layers.0.self_attn.q_proj",
        "model.layers.0.self_attn.k_proj",
        "model.layers.0.self_attn.v_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",
        "model.layers.0.mlp.up_proj",
        "model.layers.0.mlp.down_proj",
    ]


def test_fp8_modules_skip_a_shared_expert_without_width():
    hf_config = SimpleNamespace(
        model_type="qwen3_5_moe_text", layer_types=["full_attention"], shared_expert_intermediate_size=0
    )

    assert QWEN35_FP8_SPEC.fp8_modules(hf_config)[4:] == ["model.layers.0.mlp.experts"]


def test_exclusions_expand_globs_in_layer_order():
    excluded = resolve_user_provided_exclude_list(
        QWEN35_FP8_SPEC, _moe_vl_config(), ["*.layers.3.self_attn.*", "*.layers.1.mlp.experts"]
    )

    assert excluded == (
        f"{LM}.1.mlp.experts",
        f"{LM}.3.self_attn.q_proj",
        f"{LM}.3.self_attn.k_proj",
        f"{LM}.3.self_attn.v_proj",
        f"{LM}.3.self_attn.o_proj",
    )


@pytest.mark.parametrize(
    "pattern",
    [
        "*.mlp.gate",  # the router is never FP8
        "*.layers.0.mlp.experts.3.*",  # routed experts are one module per layer
        "model.layers.0.*",  # the flat text spelling, on a VL checkpoint
    ],
)
def test_exclusions_ignore_a_pattern_that_matches_no_fp8_module(pattern):
    assert resolve_user_provided_exclude_list(QWEN35_FP8_SPEC, _moe_vl_config(), [pattern]) == ()


def test_exclusions_keep_matches_when_another_pattern_matches_no_fp8_module():
    excluded = resolve_user_provided_exclude_list(
        QWEN35_FP8_SPEC,
        _moe_vl_config(),
        ["*.does_not_exist.*", "*.layers.1.mlp.experts"],
    )

    assert excluded == (f"{LM}.1.mlp.experts",)


def test_user_provided_exclude_list_extends_the_base_exclude_list():
    hf_config = _moe_vl_config()
    excluded = (f"{LM}.1.mlp.experts", f"{LM}.2.linear_attn.out_proj")

    assert engine_exclude_list(QWEN35_FP8_SPEC, hf_config, MXFP8, excluded) == QWEN35_FP8_SPEC.base_exclude_list(
        hf_config, MXFP8
    ) + [
        f"{LM}.1.mlp.experts.0.gate_proj",
        f"{LM}.1.mlp.experts.0.up_proj",
        f"{LM}.1.mlp.experts.0.down_proj",
        f"{LM}.2.linear_attn.out_proj",
    ]


@pytest.mark.parametrize("wire_format", [BLOCKWISE_FP8, MXFP8])
def test_sender_keeps_an_excluded_linear_in_model_dtype(wire_format):
    name = f"{LM}.2.linear_attn.out_proj.weight"
    tensor = torch.randn(256, 256, dtype=torch.bfloat16)

    excluded = list(
        iter_serialized_fp8_tensors(
            name, tensor, torch.bfloat16, _config(name.removesuffix(".weight"), wire_format=wire_format)
        )
    )
    assert [(n, t.dtype) for n, t in excluded] == [(name, torch.bfloat16)]
    assert torch.equal(excluded[0][1], tensor)

    quantized = list(iter_serialized_fp8_tensors(name, tensor, torch.bfloat16, _config(wire_format=wire_format)))
    assert quantized[0][1].dtype == torch.float8_e4m3fn


def test_sender_ships_excluded_routed_experts_as_the_batched_checkpoint_tensor():
    experts = f"{LM}.1.mlp.experts"
    gate_up = torch.randn(4, 256, 128, dtype=torch.bfloat16)

    emitted = list(iter_serialized_fp8_tensors(f"{experts}.gate_up_proj", gate_up, torch.bfloat16, _config(experts)))
    assert [(n, t.dtype) for n, t in emitted] == [(f"{experts}.gate_up_proj", torch.bfloat16)]

    other_layer = list(
        iter_serialized_fp8_tensors(f"{LM}.0.mlp.experts.gate_up_proj", gate_up, torch.bfloat16, _config(experts))
    )
    assert all(n.startswith(SKYRL_BATCHED_MOE_FP8_PREFIX) for n, _ in other_layer)


def test_sender_config_resolves_the_same_exclusions_as_the_engine():
    hf_config = _moe_vl_config()

    config = resolve_serialized_fp8_config(MXFP8, hf_config, ["*.layers.3.*"])

    assert config.user_provided_exclude_list == set(
        resolve_user_provided_exclude_list(QWEN35_FP8_SPEC, hf_config, ["*.layers.3.*"])
    )
    assert f"{LM}.3.mlp.experts" in config.user_provided_exclude_list
