import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.serialization import (
    LEGACY_BATCHED_MOE_PREFIX,
    SERIALIZED_BLOCKWISE_FP8,
    SERIALIZED_MXFP8,
    SERIALIZED_WEIGHT_STRATEGIES,
    get_model_quantization_spec,
    get_serialized_weight_strategy,
    iter_serialized_weight_tensors,
    model_quantization_specs,
    register_model_quantization_spec,
)


def test_builtin_serialized_weight_strategies_are_registered():
    assert tuple(SERIALIZED_WEIGHT_STRATEGIES) == (SERIALIZED_BLOCKWISE_FP8, SERIALIZED_MXFP8)
    assert [spec.name for spec in model_quantization_specs()] == ["qwen3_moe", "qwen3_5"]
    assert get_serialized_weight_strategy(SERIALIZED_BLOCKWISE_FP8).mode == SERIALIZED_BLOCKWISE_FP8
    assert get_serialized_weight_strategy(SERIALIZED_MXFP8).mode == SERIALIZED_MXFP8


def test_serialized_weight_strategy_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported serialized weight mode.*serialized_mxfp8"):
        get_serialized_weight_strategy("unknown")


def test_model_quantization_spec_registry_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate model quantization spec"):
        register_model_quantization_spec(get_model_quantization_spec("qwen3_moe"))


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


