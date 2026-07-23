import sys
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.workers.megatron._fp8_block_amax_epsilon_patch import (
    apply_fp8_block_amax_epsilon_patch,
)


def _install_fake_te_recipe(monkeypatch):
    @dataclass(frozen=True)
    class QParams:
        power_2_scale: bool
        amax_epsilon: float = 0.0
        random_hadamard_transform: bool = False
        stochastic_rounding: bool = False
        fp4_2d_quantization: bool = False

    class Float8BlockScaling:
        fp8_quant_fwd_inp = QParams(power_2_scale=False)
        fp8_quant_fwd_weight = QParams(power_2_scale=True)
        fp8_quant_bwd_grad = QParams(power_2_scale=True)

        def __init__(self, fp8_format):
            self.fp8_format = fp8_format

    transformer_engine = ModuleType("transformer_engine")
    transformer_engine.__path__ = []
    common = ModuleType("transformer_engine.common")
    common.__path__ = []
    recipe = ModuleType("transformer_engine.common.recipe")
    recipe.Float8BlockScaling = Float8BlockScaling
    recipe.Format = SimpleNamespace(E4M3="e4m3")
    transformer_engine.common = common
    common.recipe = recipe

    monkeypatch.setitem(sys.modules, "transformer_engine", transformer_engine)
    monkeypatch.setitem(sys.modules, "transformer_engine.common", common)
    monkeypatch.setitem(sys.modules, "transformer_engine.common.recipe", recipe)
    return Float8BlockScaling


def test_fp8_block_amax_patch_updates_all_qparams_and_preserves_scale_modes(monkeypatch):
    recipe_cls = _install_fake_te_recipe(monkeypatch)
    monkeypatch.setenv("NVTE_FP8_BLOCK_AMAX_EPSILON", "1e-4")

    apply_fp8_block_amax_epsilon_patch()
    apply_fp8_block_amax_epsilon_patch()

    assert recipe_cls.fp8_quant_fwd_inp.amax_epsilon == 1e-4
    assert recipe_cls.fp8_quant_fwd_weight.amax_epsilon == 1e-4
    assert recipe_cls.fp8_quant_bwd_grad.amax_epsilon == 1e-4
    assert recipe_cls.fp8_quant_fwd_inp.power_2_scale is False
    assert recipe_cls.fp8_quant_fwd_weight.power_2_scale is True


@pytest.mark.parametrize("value", ["invalid", "-0.1", "nan", "inf"])
def test_fp8_block_amax_patch_rejects_invalid_explicit_values(monkeypatch, value):
    monkeypatch.setenv("NVTE_FP8_BLOCK_AMAX_EPSILON", value)

    with pytest.raises(ValueError, match="NVTE_FP8_BLOCK_AMAX_EPSILON"):
        apply_fp8_block_amax_epsilon_patch()
