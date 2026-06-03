"""CPU unit tests for the MTP (Multi-Token Prediction) label helpers.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/utils/test_mtp_torch_utils.py
"""

import torch

from skyrl.backends.skyrl_train.utils.torch_utils import (
    build_mtp_loss_mask,
    build_mtp_next_token_labels,
)


def test_build_mtp_next_token_labels_basic():
    # labels[t] should be sequences[t + 1]; last position zeroed (no next token).
    sequences = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]])
    labels = build_mtp_next_token_labels(sequences)
    expected = torch.tensor([[11, 12, 13, 0], [21, 22, 23, 0]])
    assert torch.equal(labels, expected)
    # dtype preserved (token ids)
    assert labels.dtype == sequences.dtype


def test_build_mtp_next_token_labels_does_not_mutate_input():
    sequences = torch.tensor([[1, 2, 3]])
    seq_copy = sequences.clone()
    _ = build_mtp_next_token_labels(sequences)
    assert torch.equal(sequences, seq_copy)


def test_build_mtp_next_token_labels_no_wraparound_into_first_token():
    # The roll's wrapped value (original first token) must be overwritten with 0,
    # so the model never sees the start token as a "next token" target at the end.
    sequences = torch.tensor([[7, 8, 9]])
    labels = build_mtp_next_token_labels(sequences)
    assert labels[0, -1].item() == 0
    assert labels[0, -1].item() != sequences[0, 0].item() or sequences[0, 0].item() == 0


def test_build_mtp_loss_mask_from_attention_mask():
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    loss_mask = build_mtp_loss_mask(attention_mask)
    assert loss_mask.dtype == torch.float32
    assert torch.equal(loss_mask, attention_mask.to(torch.float32))


def test_build_mtp_loss_mask_accepts_bool_mask():
    # forward_step passes attention_mask.to(bool); the helper must still produce a float mask.
    attention_mask = torch.tensor([[True, True, False]])
    loss_mask = build_mtp_loss_mask(attention_mask)
    assert loss_mask.dtype == torch.float32
    assert torch.equal(loss_mask, torch.tensor([[1.0, 1.0, 0.0]]))
