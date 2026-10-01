import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.active_spans import (
    _build_active_span_metadata,
    _coalesced_active_spans,
    build_active_mask,
    build_packed_active_metadata,
    build_unpacked_active_metadata,
)


def test_coalesces_only_gaps_up_to_fixed_limit() -> None:
    active = torch.zeros(300, dtype=torch.bool)
    active[:10] = True
    active[138] = True  # 128 inactive rows after the first run.
    active[268] = True  # 129 inactive rows after the second run.

    assert _coalesced_active_spans(active) == ((0, 139), (268, 269))


def test_active_mask_matches_prediction_rows() -> None:
    loss_mask = torch.tensor([[0, 1, 2], [3, 0, 4]])
    expected = torch.tensor([[False, False, False, True, True], [False, False, True, False, True]])
    torch.testing.assert_close(build_active_mask(loss_mask, 3, 6), expected)


def test_unpacked_spans_match_flattened_prediction_rows() -> None:
    loss_mask = torch.zeros((2, 299))
    loss_mask[0, 50:100] = 0.25
    loss_mask[1, :20] = 1.0

    metadata = build_unpacked_active_metadata(loss_mask, 299, 300)
    assert metadata.active_spans == ((50, 100), (300, 320))
    torch.testing.assert_close(metadata.active_mask, build_active_mask(loss_mask, 299, 300))


@pytest.mark.parametrize(
    ("cp_rank", "expected"),
    [
        (0, ((1, 6),)),
        (1, ((0, 1),)),
        (2, ((7, 8),)),
        (3, ((2, 6),)),
    ],
)
def test_packed_multisequence_spans_follow_cp_chunk_order(cp_rank, expected) -> None:
    # TP1/CP4 aligns lengths 10 and 17 to 16 and 24. The second sequence
    # therefore begins at column 16 in the controller's packed row.
    loss_mask = torch.zeros((1, 39))
    loss_mask[0, [1, 2, 8]] = 1.0
    loss_mask[0, [16, 17, 26, 31]] = 1.0
    attention_mask = torch.zeros((1, 40), dtype=torch.bool)
    attention_mask[0, :10] = True
    attention_mask[0, 16:33] = True

    metadata = build_packed_active_metadata(
        loss_mask,
        39,
        40,
        attention_mask,
        sub_seq_lengths=[[10, 17]],
        tp_size=1,
        cp_size=4,
        cp_rank=cp_rank,
        fp8_enabled=False,
    )

    assert metadata.active_spans == expected


@pytest.mark.parametrize(("cp_rank", "expected"), [(0, ((0, 1),)), (1, ((0, 2),))])
def test_packed_attention_layout_uses_real_token_ordinals(cp_rank, expected) -> None:
    loss_mask = torch.zeros((1, 11))
    loss_mask[0, [2, 7, 8]] = 1.0
    attention_mask = torch.zeros((1, 12), dtype=torch.bool)
    attention_mask[0, [2, 3, 7, 8, 9]] = True

    metadata = build_packed_active_metadata(
        loss_mask,
        11,
        12,
        attention_mask,
        sub_seq_lengths=None,
        tp_size=1,
        cp_size=2,
        cp_rank=cp_rank,
        fp8_enabled=False,
    )

    assert metadata.active_spans == expected


@pytest.mark.parametrize(("cp_rank", "expected"), [(0, ((0, 9),)), (1, ((8, 9),))])
def test_packed_spans_honor_fp8_cp_alignment(cp_rank, expected) -> None:
    loss_mask = torch.zeros((1, 31))
    loss_mask[0, [0, 16, 24]] = 1.0
    attention_mask = torch.zeros((1, 32), dtype=torch.bool)
    attention_mask[0, :31] = True

    metadata = build_packed_active_metadata(
        loss_mask,
        31,
        32,
        attention_mask,
        sub_seq_lengths=[[31]],
        tp_size=1,
        cp_size=2,
        cp_rank=cp_rank,
        fp8_enabled=True,
    )

    assert metadata.active_spans == expected


def test_all_inactive_packed_mask_has_no_spans() -> None:
    assert (
        build_packed_active_metadata(
            torch.zeros((1, 7)),
            7,
            8,
            torch.ones((1, 8), dtype=torch.bool),
            sub_seq_lengths=[[1]],
            tp_size=1,
            cp_size=1,
            cp_rank=0,
            fp8_enabled=False,
        ).active_spans
        == ()
    )


def test_span_source_mask_must_agree_with_the_forward_mask() -> None:
    """The backward zeroes any active row no span reaches, so the two derivations must match."""
    forward = torch.tensor([[True, False, True, False]])

    with pytest.raises(AssertionError, match="span source mask has 3 active rows"):
        _build_active_span_metadata(forward, torch.tensor([True, False, True, True, False]))

    with pytest.raises(AssertionError, match="span source mask has 1 active rows"):
        _build_active_span_metadata(forward, torch.tensor([True, False, False, False]))

    # CP shards rows across ranks, so a local shard legitimately holds fewer.
    assert _build_active_span_metadata(forward, torch.tensor([True, False, False, False]), cp_size=2).active_spans == (
        (0, 1),
    )


def test_packed_mask_carries_the_aggregate_slab_padding() -> None:
    """TP aligns the aggregate slab, so the mask must grow past the per-sequence padding."""
    # tp=4/cp=1: per-sequence alignment is 1, aggregate alignment is 4.
    metadata = build_packed_active_metadata(
        torch.tensor([[0.0, 1.0, 0.0]]),
        num_actions=3,
        sequence_length=4,
        attention_mask=torch.tensor([[True, True, True, False]]),
        sub_seq_lengths=None,
        tp_size=4,
        cp_size=1,
        cp_rank=0,
        fp8_enabled=False,
    )

    torch.testing.assert_close(metadata.active_mask, torch.tensor([[False, True, False, False]]))
    assert metadata.active_spans == ((1, 2),)


def test_packed_mask_follows_the_fp8_recipe() -> None:
    """mxfp8 quantizes in 1x32 tiles, so it needs a different slab than blockwise."""

    def slab(recipe: str | None) -> int:
        return build_packed_active_metadata(
            torch.tensor([[0.0, 1.0, 0.0]]),
            num_actions=3,
            sequence_length=4,
            attention_mask=torch.tensor([[True, True, True, False]]),
            sub_seq_lengths=None,
            tp_size=1,
            cp_size=1,
            cp_rank=0,
            fp8_enabled=True,
            fp8_recipe=recipe,
        ).active_mask.shape[1]

    assert slab("blockwise") == 16
    assert slab("mxfp8") == 32
