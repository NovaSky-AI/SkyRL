import torch

from skyrl.backends.skyrl_train.distributed.ulysses import utils
from skyrl.backends.skyrl_train.utils.sample_support_replay import (
    aligned_sample_support_logprobs,
)


def test_ulysses_padding_preserves_trailing_metadata_dimensions(monkeypatch):
    metadata = torch.arange(2 * 5 * 3).reshape(2, 5, 3)
    rank = 0

    def slice_for_rank(tensor, dim, padding):
        return tensor.chunk(2, dim=dim)[rank]

    monkeypatch.setattr(utils, "get_ulysses_sequence_parallel_group", lambda: object())
    monkeypatch.setattr(utils, "slice_input_tensor", slice_for_rank)

    slices = []
    for rank in range(2):
        sliced, positions, attention_mask, pad_size = utils.ulysses_pad_and_slice_inputs(
            metadata,
            sp_size=2,
            input_padding_value=-1,
        )
        slices.append(sliced)

    assert sliced.shape == (2, 3, 3)
    expected = torch.cat((metadata, torch.full((2, 1, 3), -1)), dim=1)
    assert torch.equal(torch.cat(slices, dim=1), expected)
    assert positions is None
    assert attention_mask is None
    assert pad_size == 1


def test_ulysses_sample_support_matches_unsharded_values_and_gradients(monkeypatch):
    rank = 0

    def slice_for_rank(tensor, dim, padding):
        return tensor.chunk(2, dim=dim)[rank]

    monkeypatch.setattr(utils, "get_ulysses_sequence_parallel_group", lambda: object())
    monkeypatch.setattr(utils, "slice_input_tensor", slice_for_rank)

    logits = torch.randn(1, 5, 7, dtype=torch.float64, requires_grad=True)
    sampled_ids = torch.tensor([[2, 3, 4, 5, 6]])
    support_ids = torch.tensor([[[2, 0], [3, 1], [4, 2], [5, 3], [-1, -1]]], dtype=torch.int32)
    loss_mask = torch.ones((1, 5), dtype=torch.bool)
    sharded_logprobs = []
    for rank in range(2):
        local_logits, _, _, _ = utils.ulysses_pad_and_slice_inputs(logits, sp_size=2)
        local_sampled_ids, _, _, _ = utils.ulysses_pad_and_slice_inputs(sampled_ids, sp_size=2)
        local_support_ids, _, _, _ = utils.ulysses_pad_and_slice_inputs(
            support_ids,
            sp_size=2,
            input_padding_value=-1,
        )
        local_loss_mask, _, _, _ = utils.ulysses_pad_and_slice_inputs(loss_mask, sp_size=2)
        sharded_logprobs.append(
            aligned_sample_support_logprobs(
                local_logits,
                local_sampled_ids,
                local_support_ids,
                local_loss_mask,
                vocab_start_index=0,
                vocab_end_index=logits.shape[-1],
                tp_group=None,
                inference_only=False,
            )
        )

    actual = torch.cat(sharded_logprobs, dim=1)[:, : logits.shape[1]]
    actual.sum().backward()
    actual_grad = logits.grad.clone()

    reference_logits = logits.detach().clone().requires_grad_(True)
    expected = aligned_sample_support_logprobs(
        reference_logits,
        sampled_ids,
        support_ids,
        loss_mask,
        vocab_start_index=0,
        vocab_end_index=logits.shape[-1],
        tp_group=None,
        inference_only=False,
    )
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, reference_logits.grad)


def test_ulysses_synthetic_eos_uses_fixed_trajectory_capacity(monkeypatch):
    rank = 0

    def slice_for_rank(tensor, dim, padding):
        return tensor.chunk(2, dim=dim)[rank]

    monkeypatch.setattr(utils, "get_ulysses_sequence_parallel_group", lambda: object())
    monkeypatch.setattr(utils, "slice_input_tensor", slice_for_rank)

    logits = torch.randn(1, 6, 9, dtype=torch.float64, requires_grad=True)
    sampled_ids = torch.tensor([[0, 1, 2, 3, 4, 5]])
    support_ids = torch.tensor(
        [[[0, 6], [1, 7], [-1, -1], [3, 8], [4, 0], [-1, -1]]],
        dtype=torch.int32,
    )
    loss_mask = torch.ones((1, 6), dtype=torch.bool)
    trajectory_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])
    sharded_logprobs = []
    for rank in range(2):
        local = [
            utils.ulysses_pad_and_slice_inputs(tensor, sp_size=2)[0]
            for tensor in (logits, sampled_ids, support_ids, loss_mask, trajectory_ids)
        ]
        sharded_logprobs.append(
            aligned_sample_support_logprobs(
                *local[:4],
                vocab_start_index=0,
                vocab_end_index=logits.shape[-1],
                tp_group=None,
                inference_only=False,
                trajectory_ids=local[4],
                num_trajectories=2,
            )
        )

    actual = torch.cat(sharded_logprobs, dim=1)
    actual.sum().backward()
    actual_grad = logits.grad.clone()

    reference_logits = logits.detach().clone().requires_grad_(True)
    expected = aligned_sample_support_logprobs(
        reference_logits,
        sampled_ids,
        support_ids,
        loss_mask,
        vocab_start_index=0,
        vocab_end_index=logits.shape[-1],
        tp_group=None,
        inference_only=False,
        trajectory_ids=trajectory_ids,
        num_trajectories=2,
    )
    expected.sum().backward()

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, reference_logits.grad)
