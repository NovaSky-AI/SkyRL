"""Tests for Megatron optimizer dtype coercion."""

import importlib
import sys
from unittest.mock import patch

import pytest
import torch


@pytest.fixture(scope="module")
def coerce_optimizer_dtype_kwargs():
    module_name = "skyrl.backends.skyrl_train.distributed.megatron.optimizer_dtype"
    with patch.dict(sys.modules, {"megatron": None}):
        sys.modules.pop(module_name, None)
        module = importlib.import_module(module_name)
    return module.coerce_optimizer_dtype_kwargs


def test_coerces_dtype_strings_and_preserves_other_kwargs(coerce_optimizer_dtype_kwargs):
    kwargs = {
        "exp_avg_dtype": "  BF16 ",
        "exp_avg_sq_dtype": "fp8",
        "main_params_dtype": "fp16",
        "params_dtype": "float32",
        "main_grads_dtype": "bfloat16",
        "use_precision_aware_optimizer": True,
    }

    out = coerce_optimizer_dtype_kwargs(kwargs)

    assert out == {
        "exp_avg_dtype": torch.bfloat16,
        "exp_avg_sq_dtype": torch.uint8,
        "main_params_dtype": torch.float16,
        "params_dtype": torch.float32,
        "main_grads_dtype": torch.bfloat16,
        "use_precision_aware_optimizer": True,
    }
    assert kwargs["exp_avg_dtype"] == "  BF16 "


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"exp_avg_dtype": "bf17"}, "Unrecognized dtype name"),
        ({"main_params_dtype": "bf16"}, "main_params_dtype"),
    ],
)
def test_rejects_unknown_or_field_illegal_dtype_names(coerce_optimizer_dtype_kwargs, kwargs, match):
    with pytest.raises(ValueError, match=match):
        coerce_optimizer_dtype_kwargs(kwargs)


def test_none_mapping_is_empty_but_none_field_values_pass_through(coerce_optimizer_dtype_kwargs):
    assert coerce_optimizer_dtype_kwargs(None) == {}
    assert coerce_optimizer_dtype_kwargs({"main_grads_dtype": None}) == {"main_grads_dtype": None}
