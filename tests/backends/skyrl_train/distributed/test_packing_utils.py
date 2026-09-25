import pytest

from skyrl.backends.skyrl_train.distributed.megatron.packing_utils import (
    get_packing_align_size_sequence,
    get_packing_align_size_total,
    get_unpacked_seq_align_size,
    packed_segment_layout,
)

PARALLEL_CONFIGS = [
    (tp, cp, fp8, recipe)
    for tp in (1, 2, 4, 8)
    for cp in (1, 2, 4)
    for fp8, recipe in ((False, None), (True, "blockwise"), (True, "mxfp8"))
]
SEQUENCE_BATCHES = ([3], [5, 7], [1, 2, 3], [4, 9], [16, 16], [1], [127, 1], [128], [33, 31, 64])


def test_packed_alignment_uses_layout_only_without_fp8():
    assert get_packing_align_size_total(tp_size=4, cp_size=1) == 4
    assert get_packing_align_size_total(tp_size=1, cp_size=2) == 4


def test_per_sequence_layout_alignment_only_applies_with_cp():
    assert get_packing_align_size_sequence(tp_size=4, cp_size=1) == 1
    assert get_packing_align_size_sequence(tp_size=4, cp_size=2) == 16


def test_packed_alignment_adds_fp8_local_rank_multiple():
    assert get_packing_align_size_total(tp_size=4, cp_size=1, fp8_enabled=True) == 512
    assert get_packing_align_size_total(tp_size=1, cp_size=2, fp8_enabled=True) == 32
    assert get_packing_align_size_total(tp_size=2, cp_size=1, fp8_enabled=True) == 256
    assert get_packing_align_size_total(tp_size=2, cp_size=2, fp8_enabled=True) == 512


def test_unpacked_alignment_adds_fp8_multiple_only_when_enabled():
    assert get_unpacked_seq_align_size(tp_size=4) == 4
    assert get_unpacked_seq_align_size(tp_size=1, fp8_enabled=True) == 16
    assert get_unpacked_seq_align_size(tp_size=2, fp8_enabled=True) == 256
    assert get_unpacked_seq_align_size(tp_size=4, fp8_enabled=True) == 512


def test_mxfp8_recipe_aligns_to_32_token_local_shards():
    # 32*tp*cp at any TP, never the blockwise 128*tp*cp segments.
    assert get_packing_align_size_total(tp_size=1, cp_size=1, fp8_enabled=True, fp8_recipe="mxfp8") == 32
    assert get_packing_align_size_total(tp_size=1, cp_size=2, fp8_enabled=True, fp8_recipe="mxfp8") == 64
    assert get_packing_align_size_total(tp_size=2, cp_size=1, fp8_enabled=True, fp8_recipe="mxfp8") == 64
    assert get_packing_align_size_total(tp_size=2, cp_size=2, fp8_enabled=True, fp8_recipe="mxfp8") == 128
    assert get_packing_align_size_total(tp_size=4, cp_size=1, fp8_enabled=True, fp8_recipe="mxfp8") == 128
    assert get_unpacked_seq_align_size(tp_size=1, fp8_enabled=True, fp8_recipe="mxfp8") == 32
    assert get_unpacked_seq_align_size(tp_size=2, fp8_enabled=True, fp8_recipe="mxfp8") == 64
    # Non-mx recipes keep the blockwise constants.
    assert get_packing_align_size_total(tp_size=1, cp_size=1, fp8_enabled=True, fp8_recipe="blockwise") == 16
    assert get_unpacked_seq_align_size(tp_size=1, fp8_enabled=True, fp8_recipe=None) == 16


@pytest.mark.parametrize(("tp_size", "cp_size"), [(0, 1), (1, 0), (-1, 1)])
def test_packed_alignment_rejects_nonpositive_parallel_sizes(tp_size, cp_size):
    with pytest.raises(ValueError, match="must be positive"):
        get_packing_align_size_total(tp_size, cp_size, fp8_enabled=True)


def test_unpacked_alignment_rejects_nonpositive_tp_size():
    with pytest.raises(ValueError, match="must be positive"):
        get_unpacked_seq_align_size(0, fp8_enabled=True)


@pytest.mark.parametrize("tp_size,cp_size,fp8_enabled,fp8_recipe", PARALLEL_CONFIGS)
@pytest.mark.parametrize("sequence_lengths", SEQUENCE_BATCHES)
def test_packed_segment_layout_holds_both_alignments(tp_size, cp_size, fp8_enabled, fp8_recipe, sequence_lengths):
    """One source of truth for the layout every packing site used to re-derive by hand."""
    layout = packed_segment_layout(
        sequence_lengths, tp_size=tp_size, cp_size=cp_size, fp8_enabled=fp8_enabled, fp8_recipe=fp8_recipe
    )
    sequence_align = get_packing_align_size_sequence(tp_size, cp_size)
    total_align = get_packing_align_size_total(tp_size, cp_size, fp8_enabled=fp8_enabled, fp8_recipe=fp8_recipe)

    assert len(layout.padded_lengths) == len(sequence_lengths)
    # Every sub-sequence holds its own tokens, and none is truncated.
    for raw, padded in zip(sequence_lengths, layout.padded_lengths):
        assert padded >= raw
    # Sequence alignment applies to every sub-sequence; the last also carries the slab tail.
    for padded in layout.padded_lengths[:-1]:
        assert padded % sequence_align == 0
    assert layout.total % total_align == 0
    # The slab tail is attached to the final sub-sequence, never spread across all of them
    # and never appended as an extra segment: flash-attn varlen reads these boundaries.
    head = sum(raw + (-raw % sequence_align) for raw in sequence_lengths[:-1])
    assert sum(layout.padded_lengths[:-1]) == head
    assert layout.cu_seqlens_padded[0] == 0
    assert layout.cu_seqlens_padded[-1] == layout.total
    assert list(layout.cu_seqlens_padded[1:]) == [
        sum(layout.padded_lengths[: i + 1]) for i in range(len(layout.padded_lengths))
    ]


def test_packed_segment_layout_handles_an_empty_batch():
    layout = packed_segment_layout([], tp_size=4, cp_size=2)
    assert layout.padded_lengths == ()
    assert layout.cu_seqlens_padded == (0,)
    assert layout.total == 0


@pytest.mark.parametrize("bad_length", [0, -1])
def test_packed_segment_layout_rejects_nonpositive_lengths(bad_length):
    with pytest.raises(ValueError, match="must be positive"):
        packed_segment_layout([4, bad_length], tp_size=1, cp_size=1)
