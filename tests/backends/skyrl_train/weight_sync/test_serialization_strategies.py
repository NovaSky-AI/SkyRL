import pytest
import torch

from skyrl.backends.skyrl_train.quantization import (
    SERIALIZED_BLOCKWISE_FP8,
    SERIALIZED_MXFP8,
    SERIALIZED_NVFP4,
    SERIALIZED_WEIGHT_PREFIX,
    SERIALIZED_WEIGHT_STRATEGIES,
    get_quantized_model_layout,
    get_serialized_weight_strategy,
    iter_serialized_weight_tensors,
)


def test_builtin_serialized_weight_strategies_are_registered():
    assert tuple(SERIALIZED_WEIGHT_STRATEGIES) == (
        SERIALIZED_BLOCKWISE_FP8,
        SERIALIZED_MXFP8,
        SERIALIZED_NVFP4,
    )
    assert get_serialized_weight_strategy(SERIALIZED_BLOCKWISE_FP8).mode == SERIALIZED_BLOCKWISE_FP8
    assert get_serialized_weight_strategy(SERIALIZED_MXFP8).mode == SERIALIZED_MXFP8
    assert get_serialized_weight_strategy(SERIALIZED_NVFP4).mode == SERIALIZED_NVFP4


def test_serialized_weight_strategy_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported serialized weight mode.*serialized_mxfp8"):
        get_serialized_weight_strategy("unknown")


def test_qwen3_spec_and_mxfp8_strategy_emit_batched_wire_tensors():
    strategy = get_serialized_weight_strategy(SERIALIZED_MXFP8)
    layout = get_quantized_model_layout("qwen3_moe")
    name = "model.layers.0.mlp.experts.down_proj"
    weight = torch.randn(2, 64, 64, dtype=torch.bfloat16)

    emitted = dict(
        iter_serialized_weight_tensors(
            name,
            weight,
            torch.bfloat16,
            layout,
            strategy,
            model_type="qwen3_moe",
        )
    )

    weight_name = (
        f"{SERIALIZED_WEIGHT_PREFIX}{SERIALIZED_MXFP8}:model.layers.0.mlp.experts.down_proj.weight"
    )
    assert emitted[weight_name].dtype == torch.float8_e4m3fn
    assert emitted[f"{weight_name}_scale"].dtype == torch.uint8


