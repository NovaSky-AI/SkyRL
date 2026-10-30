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

    assert modules[:8] == [
        f"{LM}.0.linear_attn.in_proj_qkv",
        f"{LM}.0.linear_attn.in_proj_z",
        f"{LM}.0.linear_attn.out_proj",
        f"{LM}.0.mlp.experts.gate_up_proj",
        f"{LM}.0.mlp.experts.down_proj",
        f"{LM}.0.mlp.shared_expert.gate_proj",
        f"{LM}.0.mlp.shared_expert.up_proj",
        f"{LM}.0.mlp.shared_expert.down_proj",
    ]
    assert modules[24:28] == [f"{LM}.3.self_attn.{proj}" for proj in ("q_proj", "k_proj", "v_proj", "o_proj")]
    assert len(modules) == 3 * 8 + 9


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

    assert QWEN35_FP8_SPEC.fp8_modules(hf_config)[4:] == [
        "model.layers.0.mlp.experts.gate_up_proj",
        "model.layers.0.mlp.experts.down_proj",
    ]


@pytest.mark.parametrize("wire_format", [BLOCKWISE_FP8, MXFP8])
@pytest.mark.parametrize(
    "hf_config",
    [_moe_vl_config(), SimpleNamespace(model_type="qwen3_5_text", layer_types=["linear_attention", "full_attention"])],
    ids=["moe_vl", "dense_text"],
)
def test_every_fp8_module_is_a_tensor_the_sender_quantizes(hf_config, wire_format):
    for module in QWEN35_FP8_SPEC.fp8_modules(hf_config, wire_format):
        stacked = QWEN35_FP8_SPEC.stacked_experts(module) is not None
        assert stacked or QWEN35_FP8_SPEC.should_quantize(f"{module}.weight", (256, 256), wire_format), module


@pytest.mark.parametrize(
    "pattern",
    [
        "*.mlp.gate",  # the router is never FP8
        "*.layers.0.mlp.experts.3.*",  # the bridge stacks Qwen3.5's experts, no per-expert tensors
        "*.layers.0.mlp.experts",  # a layer's experts are its stacked tensors, not one entry
        "model.layers.0.*",  # the flat text spelling, on a VL checkpoint
    ],
)
def test_exclusions_reject_a_pattern_that_matches_no_fp8_module(pattern):
    with pytest.raises(ValueError, match="matches none"):
        resolve_user_provided_exclude_list(QWEN35_FP8_SPEC, _moe_vl_config(), BLOCKWISE_FP8, [pattern])


def test_exclusions_reject_a_non_matching_pattern_next_to_matching_ones():
    with pytest.raises(ValueError, match="'\\*.does_not_exist.\\*' matches none"):
        resolve_user_provided_exclude_list(
            QWEN35_FP8_SPEC,
            _moe_vl_config(),
            BLOCKWISE_FP8,
            ["*.layers.1.mlp.experts.*", "*.does_not_exist.*"],
        )


@pytest.mark.parametrize(
    "pattern, layer",
    [
        ("*.layers.1.mlp.experts.down_proj", 1),
        # Every layer's routed-expert down_proj, but not its gate_up_proj.
        ("*.down_proj", 0),
    ],
)
def test_exclusions_reject_part_of_a_layers_routed_experts(pattern, layer):
    with pytest.raises(ValueError, match=f"routed-expert tensors in '{LM}.{layer}.mlp.experts', but not"):
        resolve_user_provided_exclude_list(QWEN35_FP8_SPEC, _moe_vl_config(), BLOCKWISE_FP8, [pattern])


def test_exclusions_accept_a_layers_routed_experts_split_across_patterns():
    excluded = resolve_user_provided_exclude_list(
        QWEN35_FP8_SPEC,
        _moe_vl_config(),
        BLOCKWISE_FP8,
        ["*.layers.1.mlp.experts.gate_up_proj", "*.layers.1.mlp.experts.down_proj"],
    )

    assert set(excluded) == {f"{LM}.1.mlp.experts.gate_up_proj", f"{LM}.1.mlp.experts.down_proj"}


def test_user_provided_exclude_list_extends_the_base_exclude_list():
    hf_config = _moe_vl_config()
    excluded = (f"{LM}.1.mlp.experts.gate_up_proj", f"{LM}.2.linear_attn.out_proj", f"{LM}.1.mlp.experts.down_proj")

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

    config = _config(f"{experts}.gate_up_proj", f"{experts}.down_proj")

    emitted = list(iter_serialized_fp8_tensors(f"{experts}.gate_up_proj", gate_up, torch.bfloat16, config))
    assert [(n, t.dtype) for n, t in emitted] == [(f"{experts}.gate_up_proj", torch.bfloat16)]

    other_layer = list(iter_serialized_fp8_tensors(f"{LM}.0.mlp.experts.gate_up_proj", gate_up, torch.bfloat16, config))
    assert all(n.startswith(SKYRL_BATCHED_MOE_FP8_PREFIX) for n, _ in other_layer)


def test_sender_config_resolves_the_same_exclusions_as_the_engine():
    hf_config = _moe_vl_config()

    config = resolve_serialized_fp8_config(MXFP8, hf_config, ["*.layers.3.*"])

    assert config.user_provided_exclude_list == set(
        resolve_user_provided_exclude_list(QWEN35_FP8_SPEC, hf_config, MXFP8, ["*.layers.3.*"])
    )
    assert f"{LM}.3.mlp.experts.gate_up_proj" in config.user_provided_exclude_list
