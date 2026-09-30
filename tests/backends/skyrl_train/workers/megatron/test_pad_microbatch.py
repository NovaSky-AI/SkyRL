"""Tests for ``MegatronWorker._pad_microbatch_to_size``.

Token-based microbatching produces micro-batches with different row counts, and
Megatron's ``forward_backward_func`` requires a uniform micro-batch size, so every
per-sample field has to grow to the largest row count. These tests call the method
directly on an uninitialized instance -- it only reads the dict it is given -- so no
Ray actor, distributed group, or GPU is needed. (The shared session fixture in
``tests/backends/skyrl_train/conftest.py`` still calls ``ray.init()``, hence
``RAY_ADDRESS=local`` below.)

Run with:
RAY_ADDRESS=local uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/workers/megatron/test_pad_microbatch.py
"""

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import (
    TensorList,
    append_tensor_list_padding,
)

try:
    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
        MegatronWorker,
    )
except ModuleNotFoundError as e:
    if not e.name or (e.name != "megatron" and not e.name.startswith("megatron.")):
        raise
    pytest.skip(f"megatron unavailable: {e}", allow_module_level=True)


@pytest.fixture
def pad():
    worker = MegatronWorker.__new__(MegatronWorker)
    return worker._pad_microbatch_to_size


def _micro_dict(batch_size: int, seq_len: int = 8, **extra):
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    micro = {
        "sequences": torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len),
        "attention_mask": attention_mask,
        "position_ids": attention_mask.long().cumsum(-1) - 1,
        "num_actions": 4,
    }
    micro.update(extra)
    return micro


@pytest.mark.megatron
def test_pads_sub_seq_lengths_to_target_batch_size(pad):
    """``sub_seq_lengths`` is indexed by batch position, so it must grow with the rows.

    ``preprocess_packed_seqs`` raises when ``len(sub_seq_lengths)`` disagrees with
    ``input_ids.shape[0]``, and derives the segment layout from it otherwise.
    """
    sub_seq_lengths = TensorList([torch.tensor([5, 3], dtype=torch.long), torch.tensor([8], dtype=torch.long)])
    micro = _micro_dict(2, sub_seq_lengths=sub_seq_lengths)

    padded = pad(micro, 4)

    assert padded["sequences"].shape[0] == 4
    assert len(padded["sub_seq_lengths"]) == 4
    # Real rows are unchanged.
    assert padded["sub_seq_lengths"][0].tolist() == [5, 3]
    assert padded["sub_seq_lengths"][1].tolist() == [8]
    # Dummy rows describe the single valid token the padded attention_mask carries.
    assert padded["sub_seq_lengths"][2].tolist() == [1]
    assert padded["sub_seq_lengths"][3].tolist() == [1]
    assert padded["attention_mask"][2].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]


@pytest.mark.megatron
def test_padded_sub_seq_lengths_agree_with_attention_mask(pad):
    """Every row's sub-sequence lengths sum to that row's valid-token count, which is
    the invariant ``preprocess_packed_seqs`` and the packed collator share."""
    attention_mask = torch.zeros((2, 8), dtype=torch.long)
    attention_mask[0, :5] = 1
    attention_mask[1, :3] = 1
    micro = _micro_dict(2)
    micro["attention_mask"] = attention_mask
    micro["sub_seq_lengths"] = TensorList([torch.tensor([5], dtype=torch.long), torch.tensor([3], dtype=torch.long)])

    padded = pad(micro, 5)

    for row in range(5):
        assert int(padded["attention_mask"][row].sum()) == int(padded["sub_seq_lengths"][row].sum())


@pytest.mark.megatron
def test_pads_multimodal_tensor_lists_with_empty_rows(pad):
    """Dummy rows have no image, so they get the zero-row placeholder the batch builder
    already uses for text-only samples in a mixed batch."""
    pixel_values = TensorList([torch.randn(12, 6), torch.randn(4, 6)])
    image_grid_thw = TensorList([torch.tensor([[1, 3, 4]]), torch.tensor([[1, 2, 2]])])
    micro = _micro_dict(2, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    padded = pad(micro, 3)

    assert len(padded["pixel_values"]) == 3
    assert padded["pixel_values"][2].shape == (0, 6)
    assert padded["pixel_values"][2].dtype == pixel_values[0].dtype
    assert len(padded["image_grid_thw"]) == 3
    assert padded["image_grid_thw"][2].shape == (0, 3)
    # Empty rows contribute nothing to the concatenated vision input.
    assert torch.cat(padded["pixel_values"].tensors).shape == (16, 6)


@pytest.mark.megatron
def test_no_padding_needed_returns_input_unchanged(pad):
    sub_seq_lengths = TensorList([torch.tensor([5], dtype=torch.long), torch.tensor([8], dtype=torch.long)])
    micro = _micro_dict(2, sub_seq_lengths=sub_seq_lengths)

    assert pad(micro, 2) is micro
    assert len(micro["sub_seq_lengths"]) == 2


@pytest.mark.megatron
def test_padding_does_not_mutate_the_input_tensor_list(pad):
    sub_seq_lengths = TensorList([torch.tensor([5], dtype=torch.long)])
    micro = _micro_dict(1, sub_seq_lengths=sub_seq_lengths)

    padded = pad(micro, 3)

    assert len(sub_seq_lengths) == 1
    assert len(padded["sub_seq_lengths"]) == 3


@pytest.mark.megatron
def test_absent_tensor_list_fields_stay_absent(pad):
    micro = _micro_dict(2, sub_seq_lengths=None, pixel_values=None)

    padded = pad(micro, 4)

    assert padded["sub_seq_lengths"] is None
    assert padded["pixel_values"] is None
    assert padded["num_actions"] == 4


@pytest.mark.megatron
def test_append_tensor_list_padding_zero_count_is_a_no_op():
    field = TensorList([torch.tensor([5], dtype=torch.long)])

    assert append_tensor_list_padding("sub_seq_lengths", field, 0) is field
