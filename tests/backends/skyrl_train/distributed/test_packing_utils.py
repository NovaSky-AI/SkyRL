import pytest

from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (
    get_packed_seq_align_size,
    get_unpacked_seq_align_size,
    is_fp8_enabled,
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
    assert get_packed_seq_align_size(tp_size=4, cp_size=1, fp8_enabled=True) == 16
    assert get_packed_seq_align_size(tp_size=1, cp_size=2, fp8_enabled=True) == 32


def test_unpacked_alignment_adds_fp8_multiple_only_when_enabled():
    assert get_unpacked_seq_align_size(tp_size=4) == 4
    assert get_unpacked_seq_align_size(tp_size=4, fp8_enabled=True) == 16
