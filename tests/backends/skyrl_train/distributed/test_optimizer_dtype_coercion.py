"""Tests for Megatron optimizer dtype coercion."""

import sys

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.optimizer_dtype import (
    coerce_optimizer_dtype_kwargs,
)

_has_megatron = "megatron" in sys.modules or __import__("importlib").util.find_spec("megatron") is not None


class TestCoerceOptimizerDtypeKwargs:
    def _coerce(self, kwargs: dict | None) -> dict:
        return coerce_optimizer_dtype_kwargs(kwargs)

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("bf16", torch.bfloat16),
            ("bfloat16", torch.bfloat16),
            ("fp16", torch.float16),
            ("float16", torch.float16),
            ("half", torch.float16),
            ("fp32", torch.float32),
            ("float32", torch.float32),
            ("float", torch.float32),
            ("fp8", torch.uint8),
            ("float8", torch.uint8),
            ("uint8", torch.uint8),
        ],
    )
    def test_string_names_coerce_to_torch_dtype(self, name, expected):
        out = self._coerce({"exp_avg_dtype": name})
        assert out["exp_avg_dtype"] == expected
        assert isinstance(out["exp_avg_dtype"], torch.dtype)

    def test_fp8_maps_to_uint8(self):
        out = self._coerce({"exp_avg_sq_dtype": "fp8"})
        assert out["exp_avg_sq_dtype"] is torch.uint8

    def test_case_and_whitespace_insensitive(self):
        out = self._coerce({"exp_avg_dtype": "  BF16 "})
        assert out["exp_avg_dtype"] is torch.bfloat16

    def test_already_torch_dtype_passes_through(self):
        out = self._coerce({"exp_avg_dtype": torch.bfloat16})
        assert out["exp_avg_dtype"] is torch.bfloat16

    def test_main_params_dtype_accepts_fp32_and_fp16(self):
        assert self._coerce({"main_params_dtype": "fp32"})["main_params_dtype"] is torch.float32
        assert self._coerce({"main_params_dtype": "fp16"})["main_params_dtype"] is torch.float16

    @pytest.mark.parametrize("bad", ["bf16", "fp8"])
    def test_main_params_dtype_rejects_bf16_and_fp8(self, bad):
        with pytest.raises(ValueError, match="main_params_dtype"):
            self._coerce({"main_params_dtype": bad})

    @pytest.mark.parametrize(
        "name,expected", [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)]
    )
    def test_params_dtype_is_coerced_with_no_field_restriction(self, name, expected):
        out = self._coerce({"params_dtype": name})
        assert out["params_dtype"] is expected

    def test_main_grads_dtype_coerced_but_not_field_validated(self):
        out = self._coerce({"main_grads_dtype": "bf16"})
        assert out["main_grads_dtype"] is torch.bfloat16

    def test_unrecognized_dtype_name_raises(self):
        with pytest.raises(ValueError, match="Unrecognized dtype name"):
            self._coerce({"exp_avg_dtype": "bf17"})

    def test_unrelated_kwargs_pass_through_untouched(self):
        kwargs = {
            "use_precision_aware_optimizer": True,
            "optimizer_offload_fraction": 0.5,
            "overlap_cpu_optimizer_d2h_h2d": False,
            "exp_avg_dtype": "bf16",
        }
        out = self._coerce(kwargs)
        assert out["use_precision_aware_optimizer"] is True
        assert out["optimizer_offload_fraction"] == 0.5
        assert out["overlap_cpu_optimizer_d2h_h2d"] is False
        assert out["exp_avg_dtype"] is torch.bfloat16

    def test_non_string_non_dtype_dtype_value_passes_through(self):
        out = self._coerce({"main_grads_dtype": None})
        assert out["main_grads_dtype"] is None

    def test_none_kwargs_returns_empty_dict(self):
        assert self._coerce(None) == {}

    def test_input_not_mutated(self):
        kwargs = {"exp_avg_dtype": "bf16"}
        self._coerce(kwargs)
        assert kwargs["exp_avg_dtype"] == "bf16"


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
class TestInitMegatronOptimConfigDtypeCoercion:
    def test_string_dtype_kwargs_reach_optimizer_config(self):
        from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
            init_megatron_optim_config,
        )
        from skyrl.train.config import OptimizerConfig as SkyRLOptimizerConfig

        optim_config = SkyRLOptimizerConfig()
        config = init_megatron_optim_config(
            optim_config,
            {
                "use_precision_aware_optimizer": True,
                "exp_avg_dtype": "bf16",
                "exp_avg_sq_dtype": "fp8",
                "main_params_dtype": "fp32",
            },
        )
        assert config.exp_avg_dtype is torch.bfloat16
        assert config.exp_avg_sq_dtype is torch.uint8
        assert config.main_params_dtype is torch.float32

    def test_params_dtype_string_override_reaches_optimizer_config(self):
        from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
            init_megatron_optim_config,
        )
        from skyrl.train.config import OptimizerConfig as SkyRLOptimizerConfig

        config = init_megatron_optim_config(SkyRLOptimizerConfig(), {"params_dtype": "fp16"})
        assert config.params_dtype is torch.float16

    def test_default_kwargs_leave_dtypes_at_megatron_defaults(self):
        from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
            init_megatron_optim_config,
        )
        from skyrl.train.config import OptimizerConfig as SkyRLOptimizerConfig

        config = init_megatron_optim_config(SkyRLOptimizerConfig(), {})
        assert config.exp_avg_dtype is torch.float32
        assert config.exp_avg_sq_dtype is torch.float32
        assert config.main_params_dtype is torch.float32

    def test_precision_aware_off_with_nonfp32_state_fast_fails_in_megatron(self):
        from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
            init_megatron_optim_config,
        )
        from skyrl.train.config import OptimizerConfig as SkyRLOptimizerConfig

        with pytest.raises(AssertionError, match="exp_avg_dtype can only be fp32"):
            init_megatron_optim_config(
                SkyRLOptimizerConfig(),
                {"use_precision_aware_optimizer": False, "exp_avg_dtype": "bf16"},
            )
