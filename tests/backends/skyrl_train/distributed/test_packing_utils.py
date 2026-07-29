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
    assert get_packed_seq_align_size(tp_size=4, cp_size=1, fp8_enabled=True) == 512
    assert get_packed_seq_align_size(tp_size=1, cp_size=2, fp8_enabled=True) == 32
    assert get_packed_seq_align_size(tp_size=2, cp_size=1, fp8_enabled=True) == 256
    assert get_packed_seq_align_size(tp_size=2, cp_size=2, fp8_enabled=True) == 512


def test_unpacked_alignment_adds_fp8_multiple_only_when_enabled():
    assert get_unpacked_seq_align_size(tp_size=4) == 4
    assert get_unpacked_seq_align_size(tp_size=1, fp8_enabled=True) == 16
    assert get_unpacked_seq_align_size(tp_size=2, fp8_enabled=True) == 256
    assert get_unpacked_seq_align_size(tp_size=4, fp8_enabled=True) == 512


@pytest.mark.parametrize(("tp_size", "cp_size"), [(0, 1), (1, 0), (-1, 1)])
def test_packed_alignment_rejects_nonpositive_parallel_sizes(tp_size, cp_size):
    with pytest.raises(ValueError, match="must be positive"):
        get_packed_seq_align_size(tp_size, cp_size, fp8_enabled=True)


def test_unpacked_alignment_rejects_nonpositive_tp_size():
    with pytest.raises(ValueError, match="must be positive"):
        get_unpacked_seq_align_size(0, fp8_enabled=True)
