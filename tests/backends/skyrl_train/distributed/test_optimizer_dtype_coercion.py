"""CPU tests for central Megatron optimizer-state dtype coercion.

Verifies that ``*_dtype`` string values forwarded through
``MegatronConfig.optimizer_config_kwargs`` (e.g. "bf16" from YAML/Hydra) are
coerced to real ``torch.dtype`` before reaching Megatron's precision-aware
``OptimizerConfig`` / TransformerEngine FusedAdam, using Megatron-LM's own
canonical dtype mapping; that illegal values for ``main_params_dtype`` (master
weights, fp32/fp16 only) are rejected; and that unrelated kwargs pass through
untouched.

``TestCoerceOptimizerDtypeKwargs`` exercises the pure-Python (torch-only)
coercion helper, which lives in a module that does NOT import megatron-core, so
it runs unconditionally on the cheap CPU CI lane (which installs torch but not
megatron-core). ``TestInitMegatronOptimConfigDtypeCoercion`` builds a real
``OptimizerConfig`` and is therefore skipped when megatron-core is not installed
(mirroring ``test_megatron_correctness.py``). No CUDA is required by either.

uv run --isolated --extra megatron --extra dev pytest \
    tests/backends/skyrl_train/distributed/test_optimizer_dtype_coercion.py -v
"""

import sys

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.optimizer_dtype import (
    coerce_optimizer_dtype_kwargs,
)

_has_megatron = "megatron" in sys.modules or __import__("importlib").util.find_spec("megatron") is not None


class TestCoerceOptimizerDtypeKwargs:
    """``coerce_optimizer_dtype_kwargs`` maps dtype-name strings to torch.dtype.

    Runs unconditionally: the helper module is torch-only (no megatron-core).
    """

    def _coerce(self, kwargs: dict) -> dict:
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
        """Each canonical/alias dtype name maps to the right torch.dtype."""
        # exp_avg_dtype legally accepts fp32/fp16/bf16/fp8, so it can exercise all names.
        out = self._coerce({"exp_avg_dtype": name})
        assert out["exp_avg_dtype"] == expected
        assert isinstance(out["exp_avg_dtype"], torch.dtype)

    def test_fp8_maps_to_uint8(self):
        """TE represents fp8 optimizer state as uint8."""
        out = self._coerce({"exp_avg_sq_dtype": "fp8"})
        assert out["exp_avg_sq_dtype"] is torch.uint8

    def test_case_and_whitespace_insensitive(self):
        out = self._coerce({"exp_avg_dtype": "  BF16 "})
        assert out["exp_avg_dtype"] is torch.bfloat16

    def test_already_torch_dtype_passes_through(self):
        """A value already a torch.dtype is preserved as-is."""
        out = self._coerce({"exp_avg_dtype": torch.bfloat16})
        assert out["exp_avg_dtype"] is torch.bfloat16

    def test_main_params_dtype_accepts_fp32_and_fp16(self):
        """main_params_dtype (master weights) legally accepts only fp32/fp16."""
        assert self._coerce({"main_params_dtype": "fp32"})["main_params_dtype"] is torch.float32
        assert self._coerce({"main_params_dtype": "fp16"})["main_params_dtype"] is torch.float16

    @pytest.mark.parametrize("bad", ["bf16", "fp8"])
    def test_main_params_dtype_rejects_bf16_and_fp8(self, bad):
        """bf16/fp8 are illegal master-weight dtypes and must raise."""
        with pytest.raises(ValueError, match="main_params_dtype"):
            self._coerce({"main_params_dtype": bad})

    @pytest.mark.parametrize(
        "name,expected", [("bf16", torch.bfloat16), ("fp16", torch.float16), ("fp32", torch.float32)]
    )
    def test_params_dtype_is_coerced_with_no_field_restriction(self, name, expected):
        """``params_dtype`` ends in ``_dtype`` and has no legal-set entry, so it is
        coerced for any recognized alias and overrides the bf16 default that
        ``init_megatron_optim_config`` seeds (see optimizer.py)."""
        out = self._coerce({"params_dtype": name})
        assert out["params_dtype"] is expected

    def test_main_grads_dtype_coerced_but_not_field_validated(self):
        """``main_grads_dtype`` is not forwarded to FusedAdam at the pinned rev, so it
        has no legal-set row: it is coerced str->dtype but any value is accepted here,
        leaving megatron-core's ``__post_init__`` to validate it. bf16 (which a legal
        set would reject) coerces fine."""
        out = self._coerce({"main_grads_dtype": "bf16"})
        assert out["main_grads_dtype"] is torch.bfloat16

    def test_unrecognized_dtype_name_raises(self):
        with pytest.raises(ValueError, match="Unrecognized dtype name"):
            self._coerce({"exp_avg_dtype": "bf17"})

    def test_unrelated_kwargs_pass_through_untouched(self):
        """Non-``*_dtype`` keys are returned unchanged."""
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
        """A ``*_dtype`` key whose value is neither a dtype name nor torch.dtype
        (e.g. None) passes through so Megatron's own validation surfaces it."""
        out = self._coerce({"main_grads_dtype": None})
        assert out["main_grads_dtype"] is None

    def test_input_not_mutated(self):
        """The helper returns a new dict and does not mutate the input."""
        kwargs = {"exp_avg_dtype": "bf16"}
        self._coerce(kwargs)
        assert kwargs["exp_avg_dtype"] == "bf16"


@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
class TestInitMegatronOptimConfigDtypeCoercion:
    """End-to-end: ``init_megatron_optim_config`` builds a real OptimizerConfig
    with coerced dtypes from string kwargs."""

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
        """A string ``params_dtype`` override is coerced and replaces the seeded
        bf16 default in the constructed OptimizerConfig."""
        from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
            init_megatron_optim_config,
        )
        from skyrl.train.config import OptimizerConfig as SkyRLOptimizerConfig

        config = init_megatron_optim_config(SkyRLOptimizerConfig(), {"params_dtype": "fp16"})
        assert config.params_dtype is torch.float16

    def test_default_kwargs_leave_dtypes_at_megatron_defaults(self):
        """With no ``*_dtype`` overrides, OptimizerConfig keeps its fp32 defaults
        (byte-identical to prior behavior)."""
        from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
            init_megatron_optim_config,
        )
        from skyrl.train.config import OptimizerConfig as SkyRLOptimizerConfig

        config = init_megatron_optim_config(SkyRLOptimizerConfig(), {})
        assert config.exp_avg_dtype is torch.float32
        assert config.exp_avg_sq_dtype is torch.float32
        assert config.main_params_dtype is torch.float32

    def test_precision_aware_off_with_nonfp32_state_fast_fails_in_megatron(self):
        """Coercing ``exp_avg_dtype='bf16'`` passes the helper's own validation, but
        megatron-core's ``OptimizerConfig.__post_init__`` then asserts that
        exp_avg_dtype can only be fp32 when ``use_precision_aware_optimizer`` is False.

        This documents that the fast-fail is megatron's (a real AssertionError),
        not a silent mis-coercion: the helper coerces the string fine, the rejection
        happens downstream in OptimizerConfig construction.
        """
        from skyrl.backends.skyrl_train.distributed.megatron.optimizer import (
            init_megatron_optim_config,
        )
        from skyrl.train.config import OptimizerConfig as SkyRLOptimizerConfig

        with pytest.raises(AssertionError, match="exp_avg_dtype can only be fp32"):
            init_megatron_optim_config(
                SkyRLOptimizerConfig(),
                {"use_precision_aware_optimizer": False, "exp_avg_dtype": "bf16"},
            )
