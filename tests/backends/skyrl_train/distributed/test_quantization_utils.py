from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.distributed.megatron import quantization_utils
from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
    is_fp8_enabled,
    is_mxfp8_recipe,
    resolve_auto_fp8_recipe,
    resolve_auto_wire_format,
    validate_concrete_fp8_recipe,
    validate_mxfp8_gdn_tp_alignment,
    wire_to_engine_quantization,
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


@pytest.mark.parametrize(
    ("recipe", "expected"),
    [
        ("mxfp8", "mxfp8"),
        ("MXFP8 ", "mxfp8"),
        ("blockwise", "blockwise"),
        # Wire routing keys off the recipe, never the architecture: anything
        # that is not the mxfp8 recipe -- including no recipe at all -- ships
        # the blockwise wire.
        (None, "blockwise"),
        ("delayed", "blockwise"),
    ],
)
def test_resolve_auto_wire_format(recipe, expected):
    assert resolve_auto_wire_format(recipe) == expected


def test_resolve_auto_wire_format_refuses_unresolved_recipe():
    # A GPU-less driver ships fp8_recipe="auto" through unresolved. The wire
    # cannot defer to the workers (it shapes every engine's boot config), so
    # guessing blockwise here would silently mismatch an mxfp8 trainer.
    with pytest.raises(ValueError, match="fp8_weight_sync_mode"):
        resolve_auto_wire_format("auto")
    with pytest.raises(ValueError, match="fp8_weight_sync_mode"):
        resolve_auto_wire_format(" AUTO ")


def test_wire_to_engine_quantization():
    assert wire_to_engine_quantization("mxfp8") == "compressed-tensors"
    assert wire_to_engine_quantization("blockwise") == "fp8"
    # Only concrete wire formats map to an engine method; "auto" must be
    # resolved before an engine boots.
    with pytest.raises(ValueError, match="auto"):
        wire_to_engine_quantization("auto")


# --- validate_mxfp8_gdn_tp_alignment -------------------------------------------------
# Fixture = Qwen3.5-35B-A3B's real GDN geometry: in_proj_dim = 2*128*16 + 2*128*32
# + 2*32 = 12352 = 32*386, so the shard is 32-aligned only when TP divides 386.

_MXFP8_KWARGS = {"fp8": "e4m3", "fp8_recipe": "mxfp8"}


def _gdn_hf_config(nested=True):
    text = SimpleNamespace(
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    return SimpleNamespace(text_config=text) if nested else text


@pytest.mark.parametrize("tp", [1, 2])
def test_mxfp8_gdn_tp_alignment_accepts_aligned_shards(tp):
    validate_mxfp8_gdn_tp_alignment(_MXFP8_KWARGS, _gdn_hf_config(), tp)


@pytest.mark.parametrize("tp", [4, 8])
def test_mxfp8_gdn_tp_alignment_rejects_misaligned_shards(tp):
    with pytest.raises(ValueError, match="12352"):
        validate_mxfp8_gdn_tp_alignment(_MXFP8_KWARGS, _gdn_hf_config(), tp)


def test_mxfp8_gdn_tp_alignment_reads_flat_text_configs():
    with pytest.raises(ValueError, match="in_proj"):
        validate_mxfp8_gdn_tp_alignment(_MXFP8_KWARGS, _gdn_hf_config(nested=False), 4)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fp8": "e4m3", "fp8_recipe": "blockwise"},
        {"fp8": None, "fp8_recipe": "mxfp8"},
        None,
    ],
)
def test_mxfp8_gdn_tp_alignment_ignores_non_mxfp8_configs(kwargs):
    validate_mxfp8_gdn_tp_alignment(kwargs, _gdn_hf_config(), 8)


def test_mxfp8_gdn_tp_alignment_ignores_models_without_gdn():
    validate_mxfp8_gdn_tp_alignment(_MXFP8_KWARGS, SimpleNamespace(hidden_size=4096), 8)
