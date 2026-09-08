import pytest

from skyrl.backends.skyrl_train.distributed.megatron import quantization_utils
from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
    is_fp8_enabled,
    is_mxfp8_recipe,
    resolve_auto_fp8_recipe,
    validate_concrete_fp8_recipe,
)


@pytest.mark.parametrize(
    ("fp8", "expected"),
    [
        (None, False),
        ("", False),
        ("false", False),
        ("0", False),
        (False, False),
        ("hybrid", True),
        ("e4m3", True),
        (True, True),
    ],
)
def test_is_fp8_enabled(fp8, expected):
    assert is_fp8_enabled(fp8) is expected


@pytest.mark.parametrize(
    ("recipe", "expected"),
    [
        (None, False),
        ("", False),
        ("blockwise", False),
        ("delayed", False),
        ("mxfp8", True),
        (" MXFP8 ", True),
    ],
)
def test_is_mxfp8_recipe(recipe, expected):
    assert is_mxfp8_recipe(recipe) is expected


def test_resolve_auto_fp8_recipe_picks_mxfp8_on_blackwell(monkeypatch):
    monkeypatch.setattr(quantization_utils, "has_visible_cuda_device", lambda: True)
    monkeypatch.setattr(quantization_utils, "is_blackwell_or_newer", lambda: True)
    kwargs = {"fp8": "e4m3", "fp8_recipe": "auto"}
    assert resolve_auto_fp8_recipe(kwargs) == "mxfp8"
    assert kwargs["fp8_recipe"] == "mxfp8"


def test_resolve_auto_fp8_recipe_picks_blockwise_on_hopper(monkeypatch):
    monkeypatch.setattr(quantization_utils, "has_visible_cuda_device", lambda: True)
    monkeypatch.setattr(quantization_utils, "is_blackwell_or_newer", lambda: False)
    kwargs = {"fp8": "e4m3", "fp8_recipe": "AUTO"}
    assert resolve_auto_fp8_recipe(kwargs) == "blockwise"
    assert kwargs["fp8_recipe"] == "blockwise"


def test_resolve_auto_fp8_recipe_defers_without_cuda(monkeypatch):
    """A GPU-less driver must not guess the workers' architecture."""
    monkeypatch.setattr(quantization_utils, "has_visible_cuda_device", lambda: False)
    kwargs = {"fp8": "e4m3", "fp8_recipe": "auto"}
    assert resolve_auto_fp8_recipe(kwargs) == "auto"
    assert kwargs["fp8_recipe"] == "auto"


def test_validate_concrete_fp8_recipe_ignores_non_mxfp8():
    validate_concrete_fp8_recipe({"fp8_recipe": "blockwise", "fp8_param": True})
    validate_concrete_fp8_recipe({"fp8_recipe": "auto"})
    validate_concrete_fp8_recipe({})
    validate_concrete_fp8_recipe(None)


def test_validate_concrete_fp8_recipe_rejects_mxfp8_before_blackwell(monkeypatch):
    monkeypatch.setattr(quantization_utils, "has_visible_cuda_device", lambda: True)
    monkeypatch.setattr(quantization_utils, "is_blackwell_or_newer", lambda: False)
    with pytest.raises(ValueError, match="requires SM100"):
        validate_concrete_fp8_recipe({"fp8_recipe": "mxfp8"})


def test_validate_concrete_fp8_recipe_rejects_mxfp8_with_fp8_param(monkeypatch):
    # Device-independent: must fire even on a GPU-less process.
    monkeypatch.setattr(quantization_utils, "has_visible_cuda_device", lambda: False)
    with pytest.raises(ValueError, match="fp8_param"):
        validate_concrete_fp8_recipe({"fp8_recipe": "mxfp8", "fp8_param": True})


def test_resolve_auto_fp8_recipe_passes_explicit_values_through(monkeypatch):
    monkeypatch.setattr(quantization_utils, "is_blackwell_or_newer", lambda: False)
    kwargs = {"fp8_recipe": "blockwise"}
    assert resolve_auto_fp8_recipe(kwargs) == "blockwise"
    assert kwargs["fp8_recipe"] == "blockwise"
    assert resolve_auto_fp8_recipe({}) is None
    assert resolve_auto_fp8_recipe(None) is None


def test_resolve_auto_fp8_recipe_warns_for_emulated_blockwise_on_blackwell(monkeypatch):
    monkeypatch.setattr(quantization_utils, "is_blackwell_or_newer", lambda: True)
    warnings = []
    monkeypatch.setattr(
        quantization_utils.logger, "warning", lambda msg, *args, **kw: warnings.append(msg.format(*args))
    )
    kwargs = {"fp8_recipe": "blockwise"}
    assert resolve_auto_fp8_recipe(kwargs) == "blockwise"
    assert kwargs["fp8_recipe"] == "blockwise"
    assert any("emulated" in w for w in warnings)
    # The native recipe stays silent.
    warnings.clear()
    assert resolve_auto_fp8_recipe({"fp8_recipe": "mxfp8"}) == "mxfp8"
    assert not warnings
