"""CPU tests for ``MegatronWorker._pad_microbatch_to_size``.

Token-based microbatching yields micro-batches with different row counts, and Megatron's
``forward_backward_func`` needs a uniform micro-batch size, so every per-sample field has
to grow to the largest row count. The method only reads the dict it is given, so it is
called on an uninitialized instance: no Ray actor, process group, or GPU is needed.
"""

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TensorList

pytestmark = pytest.mark.megatron

try:
    from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
        MegatronWorker,
    )
except ModuleNotFoundError as e:
    if not e.name or (e.name != "megatron" and not e.name.startswith("megatron.")):
        raise
    pytest.skip(f"megatron worker unavailable: {e}", allow_module_level=True)


@pytest.fixture
def pad():
    return MegatronWorker.__new__(MegatronWorker)._pad_microbatch_to_size


def _micro_dict(batch_size: int, seq_len: int = 8, **extra):
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    micro = {
        "sequences": torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len),
        "attention_mask": attention_mask,
        "position_ids": attention_mask.cumsum(-1) - 1,
        "num_actions": 4,
    }
    micro.update(extra)
    return micro


@pytest.mark.parametrize("dtype", [torch.long, torch.int32])
def test_pads_sub_seq_lengths_with_one_token_rows(pad, dtype):
    """``preprocess_packed_seqs`` raises when ``len(sub_seq_lengths)`` disagrees with the batch size."""
    sub_seq_lengths = TensorList([torch.tensor([5, 3], dtype=dtype), torch.tensor([8], dtype=dtype)])

    padded = pad(_micro_dict(2, sub_seq_lengths=sub_seq_lengths), 4)

    assert padded["sequences"].shape[0] == 4
    assert isinstance(padded["sub_seq_lengths"], TensorList)
    assert [row.tolist() for row in padded["sub_seq_lengths"]] == [[5, 3], [8], [1], [1]]
    assert all(row.dtype == dtype for row in padded["sub_seq_lengths"])
    assert padded["attention_mask"][2:].tolist() == [[1, 0, 0, 0, 0, 0, 0, 0]] * 2


def test_padded_sub_seq_lengths_agree_with_attention_mask(pad):
    """Each row's sub-sequence lengths sum to its valid-token count, the invariant packing relies on."""
    attention_mask = torch.zeros((2, 8), dtype=torch.long)
    attention_mask[0, :5] = 1
    attention_mask[1, :3] = 1
    micro = _micro_dict(2)
    micro["attention_mask"] = attention_mask
    micro["sub_seq_lengths"] = TensorList([torch.tensor([2, 3]), torch.tensor([3])])

    padded = pad(micro, 5)

    for row in range(5):
        assert int(padded["attention_mask"][row].sum()) == int(padded["sub_seq_lengths"][row].sum())


def test_pads_multimodal_tensor_lists_with_empty_rows(pad):
    """Dummy rows carry no image, so they contribute no rows to the concatenated vision input."""
    pixel_values = TensorList([torch.randn(12, 6, dtype=torch.bfloat16), torch.randn(4, 6, dtype=torch.bfloat16)])
    image_grid_thw = TensorList([torch.tensor([[1, 3, 4]]), torch.tensor([[1, 2, 2]])])

    padded = pad(_micro_dict(2, pixel_values=pixel_values, image_grid_thw=image_grid_thw), 3)

    assert len(padded["pixel_values"]) == 3
    assert padded["pixel_values"][2].shape == (0, 6)
    assert padded["pixel_values"][2].dtype == torch.bfloat16
    assert len(padded["image_grid_thw"]) == 3
    assert padded["image_grid_thw"][2].shape == (0, 3)
    assert torch.cat(padded["pixel_values"].tensors).shape == (16, 6)
    assert torch.cat(padded["image_grid_thw"].tensors).tolist() == [[1, 3, 4], [1, 2, 2]]


def test_no_padding_needed_returns_input_unchanged(pad):
    sub_seq_lengths = TensorList([torch.tensor([5]), torch.tensor([8])])
    micro = _micro_dict(2, sub_seq_lengths=sub_seq_lengths)

    assert pad(micro, 2) is micro
    assert micro["sub_seq_lengths"] is sub_seq_lengths


def test_padding_does_not_mutate_input_tensor_lists(pad):
    sub_seq_lengths = TensorList([torch.tensor([5])])
    pixel_values = TensorList([torch.randn(2, 6)])

    padded = pad(_micro_dict(1, sub_seq_lengths=sub_seq_lengths, pixel_values=pixel_values), 3)

    assert len(sub_seq_lengths) == 1
    assert len(pixel_values) == 1
    assert len(padded["sub_seq_lengths"]) == 3
    assert len(padded["pixel_values"]) == 3


def test_absent_tensor_list_fields_stay_absent(pad):
    padded = pad(_micro_dict(2, sub_seq_lengths=None, pixel_values=None, image_grid_thw=None), 4)

    assert padded["sub_seq_lengths"] is None
    assert padded["pixel_values"] is None
    assert padded["image_grid_thw"] is None
    assert padded["num_actions"] == 4
