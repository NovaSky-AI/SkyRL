"""CPU tests for the hard-label MTP draft loss (cross-entropy on the generated tokens): the target shift in the packed
log-prob helpers (CP=1 and CP=2 layouts), its alignment to the action tokens, and the pipeline stages that lay out
the token ids.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/mtp/test_mtp_draft_loss.py
"""

from unittest.mock import patch

import pytest
import torch

from skyrl.backends.skyrl_train.mtp.draft_loss import (
    distribution_valid_lengths,
    distribution_valid_mask,
    mtp_action_aligned,
    mtp_draft_token_counts,
    segment_real_lengths,
)
from skyrl.backends.skyrl_train.mtp.hidden_capture import stage_needs_input_ids
from skyrl.train.config.config import MTPLossType

# CPU CI also runs these tests without the megatron extra; the tests that need it skip.
try:
    import megatron.core.parallel_state as mpu

    from skyrl.backends.skyrl_train.distributed.megatron import model_utils

    _has_megatron = True
except ImportError:
    _has_megatron = False

SEQ = 8
BASE = 100  # token ids are BASE + position, so a scored id names the position it came from


def _scored_ids(target_shift: int, cp_size: int = 1, cp_rank: int = 0) -> torch.Tensor:
    """The target id each local packed position scores, via a fused op that returns its targets."""
    seen = {}

    def fused_apply(backend, hidden, weight, rolled_targets, *args, **kwargs):
        seen["targets"] = rolled_targets.clone()
        return rolled_targets.to(torch.float32)

    target = (BASE + torch.arange(SEQ)).unsqueeze(0)
    with (
        patch.object(model_utils.torch.distributed, "get_world_size", return_value=cp_size),
        patch.object(model_utils.torch.distributed, "get_rank", return_value=cp_rank),
        patch.object(model_utils, "_fused_lm_head_logprob_apply", side_effect=fused_apply),
        patch.object(model_utils, "allgather_cp_sharded_packed_tensor", side_effect=lambda t, *a: t),
    ):
        model_utils.from_parallel_hidden_to_logprobs_packed_sequences(
            torch.zeros((1, SEQ // cp_size, 4)),
            torch.zeros((16, 4)),
            target,
            torch.tensor([0, SEQ]),
            unpacked_seqlen=SEQ,
            vocab_start_index=0,
            vocab_end_index=16,
            group=object(),
            cp_group=object() if cp_size > 1 else None,
            target_shift=target_shift,
        )
    return seen["targets"].squeeze(0)


@pytest.mark.megatron
@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("target_shift", [1, 2, 3])
def test_packed_target_shift_scores_the_token_that_many_positions_ahead(cp_size, cp_rank, target_shift):
    # target_shift=0 maps each local position to its global position under this CP layout.
    positions = _scored_ids(0, cp_size, cp_rank) - BASE
    expected = BASE + torch.remainder(positions + target_shift, SEQ)
    torch.testing.assert_close(_scored_ids(target_shift, cp_size, cp_rank), expected)


def _token_lp_scoring(target_shift: int, seq: int = SEQ) -> torch.Tensor:
    """A [1, seq - 1] stand-in for the unpacked MTP log-probs: column s holds the index it scores."""
    return (torch.arange(seq - 1, dtype=torch.float32) + target_shift).unsqueeze(0)


@pytest.mark.parametrize("target_shift", [1, 2, 3])
def test_alignment_puts_each_action_tokens_score_in_its_column(target_shift):
    num_actions = 3
    lp, valid = mtp_action_aligned(
        _token_lp_scoring(target_shift),
        num_actions,
        target_shift,
        loss_mask=torch.ones(1, num_actions),
        attention_mask=torch.ones(1, SEQ),
    )
    # Action i is sequence token SEQ - num_actions + i, whatever the shift.
    torch.testing.assert_close(lp, torch.tensor([[5.0, 6.0, 7.0]]))
    torch.testing.assert_close(valid, torch.ones(1, num_actions))


def test_loss_mask_and_padding_invalidate_positions():
    target_shift = 2
    num_actions = 3
    loss_mask = torch.tensor([[1.0, 0.0, 1.0]])
    # Left padding: tokens 0-3 are padding, so the head position for action 0 (token 5, position 3) is not real.
    attention_mask = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    lp, valid = mtp_action_aligned(
        _token_lp_scoring(target_shift), num_actions, target_shift, loss_mask, attention_mask
    )
    torch.testing.assert_close(valid, torch.tensor([[0.0, 0.0, 1.0]]))
    torch.testing.assert_close(lp, torch.tensor([[0.0, 0.0, 7.0]]))


def test_targets_without_a_head_position_are_invalid():
    # Five actions in an 8-token sequence at shift 4: action 0 (token 3) would need head position -1.
    lp, valid = mtp_action_aligned(_token_lp_scoring(4), 5, 4, torch.ones(1, 5), torch.ones(1, SEQ))
    torch.testing.assert_close(valid, torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0]]))
    torch.testing.assert_close(lp, torch.tensor([[0.0, 4.0, 5.0, 6.0, 7.0]]))


@pytest.mark.megatron
@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
@pytest.mark.parametrize(
    "first, last, mtp, expected",
    [(True, False, False, True), (False, True, False, False), (False, True, True, True), (False, False, True, False)],
)
def test_the_last_pipeline_stage_lays_out_token_ids_for_the_mtp_block(monkeypatch, first, last, mtp, expected):
    monkeypatch.setattr(mpu, "is_pipeline_first_stage", lambda ignore_virtual: first)
    monkeypatch.setattr(mpu, "is_pipeline_last_stage", lambda ignore_virtual: last)
    assert stage_needs_input_ids(mtp) == expected


def test_distribution_valid_mask_keeps_each_segments_leading_valid_tokens():
    """The count sums distribution_valid_lengths; the loss masks that many tokens from each padded segment start."""
    attention_mask = torch.tensor([[0, 1, 1, 1, 0, 0], [1, 0, 1, 1, 1, 1]])
    assert segment_real_lengths(attention_mask).tolist() == [3, 5]
    lengths = segment_real_lengths(attention_mask, [torch.tensor([3]), [1, 0, 4]])  # a 1-token and an empty segment
    assert lengths.tolist() == [3, 1, 0, 4]
    padded = torch.tensor([0, 4, 8, 8, 12])
    expected_mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]], dtype=torch.bool)
    assert torch.equal(distribution_valid_mask(lengths, 0, padded), expected_mask)
    for depth, expected in enumerate([[2, 0, 0, 3], [1, 0, 0, 2], [0, 0, 0, 1]]):
        assert distribution_valid_lengths(lengths, depth).tolist() == expected
        assert distribution_valid_mask(lengths, depth, padded).sum() == sum(expected)


@pytest.mark.megatron
@pytest.mark.skipif(not _has_megatron, reason="megatron-core not installed")
@pytest.mark.parametrize(
    "loss_type,packed,expected",
    [
        (MTPLossType.HARD_LABEL, True, [4, 2]),
        (MTPLossType.DISTRIBUTION, False, [6, 4]),
        (MTPLossType.DISTRIBUTION, True, [6, 4]),
    ],
)
def test_draft_token_counts_cover_every_microbatch_and_dp_rank(monkeypatch, loss_type, packed, expected):
    """The per-depth denominators every microbatch's draft loss shares, so the losses add up to the token mean."""
    micro_batches = [
        # 4 real tokens (positions 4-7), 3 actions. Hard label, depth 0 (shift 2): action 0's head position (3)
        # is padding, so 2 count; depth 1: 1. Distribution: every position with a real target, 4 - (depth + 1).
        dict(attention_mask=torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]]), loss_mask=torch.ones(1, 3), num_actions=3),
        # A padding microbatch, as token-based batching builds it: one real token, nothing to score.
        dict(attention_mask=torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]), loss_mask=torch.zeros(1, 3), num_actions=3),
    ]
    dp_size = 2  # every DP rank holding the same counts
    monkeypatch.setattr(mpu, "get_data_parallel_group", lambda with_context_parallel: None)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda t, group: t.mul_(dp_size))
    counts = mtp_draft_token_counts(micro_batches, 2, loss_type=loss_type, packed=packed)
    torch.testing.assert_close(counts, torch.tensor(expected, dtype=torch.float32))
