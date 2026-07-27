from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.serialization import (
    LEGACY_BATCHED_MOE_PREFIX,
    SERIALIZED_BLOCKWISE_FP8,
    SERIALIZED_MXFP8,
    get_model_quantization_spec,
    get_serialized_weight_strategy,
    get_serialized_weight_sync_mode,
    iter_serialized_weight_tensors,
    model_quantization_specs,
    parse_serialized_wire_name,
    register_model_quantization_spec,
    register_serialized_weight_strategy,
    resolve_serialized_receiver_target,
    serialized_weight_modes,
)


def test_builtin_serialized_weight_strategies_are_registered():
    assert serialized_weight_modes() == (SERIALIZED_BLOCKWISE_FP8, SERIALIZED_MXFP8)
    assert [spec.name for spec in model_quantization_specs()] == ["qwen3_moe", "qwen3_5"]
    assert get_serialized_weight_strategy(SERIALIZED_BLOCKWISE_FP8).mode == SERIALIZED_BLOCKWISE_FP8
    assert get_serialized_weight_strategy(SERIALIZED_MXFP8).mode == SERIALIZED_MXFP8


def test_serialized_weight_strategy_registry_rejects_duplicates_and_unknown_modes():
    with pytest.raises(ValueError, match="Duplicate serialized weight strategy"):
        register_serialized_weight_strategy(SERIALIZED_MXFP8, lambda: get_serialized_weight_strategy(SERIALIZED_MXFP8))
    with pytest.raises(ValueError, match="Unsupported serialized weight mode.*serialized_mxfp8"):
        get_serialized_weight_strategy("unknown")


def test_model_quantization_spec_registry_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate model quantization spec"):
        register_model_quantization_spec(get_model_quantization_spec("qwen3_moe"))


def test_serialized_weight_sync_mode_supports_legacy_alias():
    assert (
        get_serialized_weight_sync_mode(SimpleNamespace(serialized_weight_sync_mode=None, fp8_weight_sync_mode="x"))
        == "x"
    )
    assert (
        get_serialized_weight_sync_mode(SimpleNamespace(serialized_weight_sync_mode="x", fp8_weight_sync_mode=None))
        == "x"
    )
    with pytest.raises(ValueError, match="Set only one"):
        get_serialized_weight_sync_mode(SimpleNamespace(serialized_weight_sync_mode="x", fp8_weight_sync_mode="x"))


def test_qwen3_spec_and_mxfp8_strategy_emit_batched_wire_tensors():
    model_spec = get_model_quantization_spec("qwen3_moe")
    strategy = get_serialized_weight_strategy(SERIALIZED_MXFP8)
    name = "model.layers.0.mlp.experts.down_proj"
    weight = torch.randn(2, 64, 64, dtype=torch.bfloat16)

    emitted = dict(
        iter_serialized_weight_tensors(
            name,
            weight,
            torch.bfloat16,
            model_spec,
            strategy,
            model_type="qwen3_moe",
        )
    )

    weight_name = f"{LEGACY_BATCHED_MOE_PREFIX}model.layers.0.mlp.experts.down_proj.weight"
    assert emitted[weight_name].dtype == torch.float8_e4m3fn
    assert emitted[f"{weight_name}_scale"].dtype == torch.uint8


@pytest.mark.parametrize(
    ("checkpoint_name", "parameter_name", "shard_id"),
    [
        (
            "model.layers.0.mlp.experts.gate_proj.weight",
            "model.layers.0.mlp.experts.w13_weight",
            "w1",
        ),
        (
            "model.layers.0.mlp.experts.up_proj.weight_scale_inv",
            "model.layers.0.mlp.experts.w13_weight_scale_inv",
            "w3",
        ),
        (
            "model.layers.0.mlp.experts.down_proj.weight_scale",
            "model.layers.0.mlp.experts.w2_weight_scale",
            "w2",
        ),
    ],
)
def test_strategy_receiver_targets_match_existing_fused_moe_parameters(
    checkpoint_name,
    parameter_name,
    shard_id,
):
    target = resolve_serialized_receiver_target(None, checkpoint_name)
    assert target is not None
    assert target.parameter_name == parameter_name
    assert target.shard_id == shard_id


def test_serialized_wire_parser_preserves_legacy_names_and_supports_qualified_modes():
    checkpoint_name = "model.layers.0.mlp.experts.gate_proj.weight"
    assert parse_serialized_wire_name(f"{LEGACY_BATCHED_MOE_PREFIX}{checkpoint_name}") == (None, checkpoint_name)
    assert parse_serialized_wire_name(f"__skyrl_serialized__:serialized_mxfp8:{checkpoint_name}") == (
        SERIALIZED_MXFP8,
        checkpoint_name,
    )
    assert parse_serialized_wire_name(checkpoint_name) is None
