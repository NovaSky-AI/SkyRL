"""Token-aligned side channels through an Ulysses sequence-parallel partition.

Ulysses splits the sequence axis across ranks, so a per-token channel has to take the same
partition as the tokens -- including the pad to a multiple of the SP degree, where a channel
whose ``0`` is a real value needs its own sentinel. The all-to-all itself needs a process
group, so these tests shard by monkeypatching ``slice_input_tensor`` and drive the replay core
once per rank, which is exactly the split the real forward performs.

Run with:
uv run --isolated --extra dev --extra skyrl-train pytest \
    tests/backends/skyrl_train/distributed/test_ulysses_token_metadata.py
"""

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.ulysses import utils
from skyrl.backends.skyrl_train.utils.packed_tensor import (
    PackedTensor,
    cu_seqlens_from_lengths,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_NO_ROW,
    SAMPLE_SUPPORT_TORCH_DTYPE,
)
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    score_aligned_sample_support,
)

SP_SIZE = 2
VOCAB = 9


@pytest.fixture
def sharded(monkeypatch):
    """Slice for a chosen rank without a process group. Returns a rank setter."""
    state = {"rank": 0}

    def slice_for_rank(tensor, dim, padding=True, group=None):
        return tensor.chunk(SP_SIZE, dim=dim)[state["rank"]]

    monkeypatch.setattr(utils, "get_ulysses_sequence_parallel_group", lambda: object())
    monkeypatch.setattr(utils, "slice_input_tensor", slice_for_rank)
    return state


def _support(rows: list[list[int]], segment_lengths: list[int]) -> PackedTensor:
    return PackedTensor(
        torch.tensor(rows, dtype=SAMPLE_SUPPORT_TORCH_DTYPE),
        cu_seqlens_from_lengths(segment_lengths),
    )


def _score(logits, sampled_ids, row_ids, loss_mask, support, **kwargs):
    return score_aligned_sample_support(
        logits,
        sampled_ids,
        row_ids,
        loss_mask,
        support,
        vocab_start_index=0,
        vocab_end_index=logits.shape[-1],
        tp_group=None,
        **kwargs,
    )


def test_padding_uses_the_requested_fill_along_the_sequence_dimension(sharded):
    """``0`` is a valid row id, so a row-id channel cannot pad with the token-id default."""
    row_ids = torch.arange(5, dtype=torch.long).unsqueeze(0)

    shards = []
    for rank in range(SP_SIZE):
        sharded["rank"] = rank
        shard, positions, attention_mask, pad_size = utils.ulysses_pad_and_slice_inputs(
            row_ids,
            sp_size=SP_SIZE,
            input_padding_value=SAMPLE_SUPPORT_NO_ROW,
        )
        shards.append(shard)

    assert pad_size == 1
    assert positions is None and attention_mask is None
    assert torch.cat(shards, dim=1).tolist() == [[0, 1, 2, 3, 4, SAMPLE_SUPPORT_NO_ROW]]
    # The default is still a zero fill, which is what token ids want.
    sharded["rank"] = 1
    assert utils.ulysses_pad_and_slice_inputs(row_ids, sp_size=SP_SIZE)[0].tolist() == [[3, 4, 0]]


def test_padding_preserves_trailing_channel_dimensions(sharded):
    """Padding goes along the sequence dim, not the last one, so ``[b, s, k]`` survives."""
    channel = torch.arange(2 * 5 * 3).reshape(2, 5, 3)

    shards = []
    for rank in range(SP_SIZE):
        sharded["rank"] = rank
        shards.append(utils.ulysses_pad_and_slice_inputs(channel, sp_size=SP_SIZE, input_padding_value=-1)[0])

    assert shards[0].shape == (2, 3, 3)
    expected = torch.cat((channel, torch.full((2, 1, 3), -1)), dim=1)
    assert torch.equal(torch.cat(shards, dim=1), expected)


def test_sharded_support_scores_match_the_unsharded_pass(sharded):
    """Values and gradients, over a sequence whose length is not a multiple of the SP degree."""
    logits = torch.randn(1, 5, VOCAB, dtype=torch.float64, requires_grad=True)
    sampled_ids = torch.tensor([[2, 3, 4, 5, 6]])
    row_ids = torch.tensor([[0, 1, 2, 3, 4]])
    loss_mask = torch.ones((1, 5), dtype=torch.bool)
    support = _support([[2, 0], [3, 1], [4, 2], [5, 3], [6, 4]], [5])

    shards = []
    for rank in range(SP_SIZE):
        sharded["rank"] = rank
        local = [
            utils.ulysses_pad_and_slice_inputs(tensor, sp_size=SP_SIZE, input_padding_value=fill)[0]
            for tensor, fill in (
                (logits, 0),
                (sampled_ids, 0),
                (row_ids, SAMPLE_SUPPORT_NO_ROW),
                (loss_mask, 0),
            )
        ]
        shards.append(_score(*local, support).logprobs)

    actual = torch.cat(shards, dim=1)[:, : logits.shape[1]]
    actual.sum().backward()

    reference_logits = logits.detach().clone().requires_grad_(True)
    expected = _score(
        reference_logits,
        sampled_ids,
        row_ids,
        loss_mask,
        support,
    ).logprobs
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(logits.grad, reference_logits.grad)
