import pytest

from skyrl.backends.skyrl_train.distributed.megatron import packing_utils
from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (
    get_packed_seq_align_size,
    get_unpacked_seq_align_size,
    is_fp8_enabled,
    resolve_auto_fp8_recipe,
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


def test_packed_alignment_uses_layout_only_without_fp8():
    assert get_packed_seq_align_size(tp_size=4, cp_size=1) == 4
    assert get_packed_seq_align_size(tp_size=1, cp_size=2) == 4


def test_packed_alignment_adds_fp8_local_rank_multiple():
    assert get_packed_seq_align_size(tp_size=4, cp_size=1, fp8_enabled=True) == 512
    assert get_packed_seq_align_size(tp_size=1, cp_size=2, fp8_enabled=True) == 32
    assert get_packed_seq_align_size(tp_size=2, cp_size=1, fp8_enabled=True) == 256
    assert get_packed_seq_align_size(tp_size=2, cp_size=2, fp8_enabled=True) == 512


def test_unpacked_alignment_adds_fp8_multiple_only_when_enabled():
    assert get_unpacked_seq_align_size(tp_size=4) == 4
    assert get_unpacked_seq_align_size(tp_size=1, fp8_enabled=True) == 16
    assert get_unpacked_seq_align_size(tp_size=2, fp8_enabled=True) == 256
    assert get_unpacked_seq_align_size(tp_size=4, fp8_enabled=True) == 512


def test_mxfp8_recipe_widens_tp1_alignment_to_32():
    # MXFP8 quantizes in 1x32 tiles (TE asserts dims % 32 == 0), so the
    # TP=1 slab grows from 16 to 32; the tp>1 128-token segments are
    # already 32-divisible and must not change.
    assert get_packed_seq_align_size(tp_size=1, cp_size=1, fp8_enabled=True, fp8_recipe="mxfp8") == 32
    assert get_packed_seq_align_size(tp_size=1, cp_size=2, fp8_enabled=True, fp8_recipe="mxfp8") == 64
    assert get_packed_seq_align_size(tp_size=2, cp_size=1, fp8_enabled=True, fp8_recipe="mxfp8") == 256
    assert get_packed_seq_align_size(tp_size=2, cp_size=2, fp8_enabled=True, fp8_recipe="mxfp8") == 512
    assert get_unpacked_seq_align_size(tp_size=1, fp8_enabled=True, fp8_recipe="mxfp8") == 32
    assert get_unpacked_seq_align_size(tp_size=2, fp8_enabled=True, fp8_recipe="mxfp8") == 256
    # Non-mx recipes keep the blockwise constants.
    assert get_packed_seq_align_size(tp_size=1, cp_size=1, fp8_enabled=True, fp8_recipe="blockwise") == 16
    assert get_unpacked_seq_align_size(tp_size=1, fp8_enabled=True, fp8_recipe=None) == 16


@pytest.mark.parametrize(("tp_size", "cp_size"), [(0, 1), (1, 0), (-1, 1)])
def test_packed_alignment_rejects_nonpositive_parallel_sizes(tp_size, cp_size):
    with pytest.raises(ValueError, match="must be positive"):
        get_packed_seq_align_size(tp_size, cp_size, fp8_enabled=True)


def test_unpacked_alignment_rejects_nonpositive_tp_size():
    with pytest.raises(ValueError, match="must be positive"):
        get_unpacked_seq_align_size(0, fp8_enabled=True)


def test_resolve_auto_fp8_recipe_picks_mxfp8_on_blackwell(monkeypatch):
    monkeypatch.setattr(packing_utils, "is_blackwell_or_newer", lambda: True)
    kwargs = {"fp8": "e4m3", "fp8_recipe": "auto"}
    assert resolve_auto_fp8_recipe(kwargs) == "mxfp8"
    assert kwargs["fp8_recipe"] == "mxfp8"


def test_resolve_auto_fp8_recipe_picks_blockwise_on_hopper(monkeypatch):
    monkeypatch.setattr(packing_utils, "is_blackwell_or_newer", lambda: False)
    kwargs = {"fp8": "e4m3", "fp8_recipe": "AUTO"}
    assert resolve_auto_fp8_recipe(kwargs) == "blockwise"
    assert kwargs["fp8_recipe"] == "blockwise"


def test_resolve_auto_fp8_recipe_passes_explicit_values_through():
    kwargs = {"fp8_recipe": "blockwise"}
    assert resolve_auto_fp8_recipe(kwargs) == "blockwise"
    assert kwargs["fp8_recipe"] == "blockwise"
    assert resolve_auto_fp8_recipe({}) is None
    assert resolve_auto_fp8_recipe(None) is None
