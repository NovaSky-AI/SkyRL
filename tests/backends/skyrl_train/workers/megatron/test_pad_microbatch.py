"""Tests for ``HostMicrobatch.padded``.

Token-based microbatching produces micro-batches with different row counts, and
Megatron's ``forward_backward_func`` requires a uniform micro-batch size, so every
per-sample field has to grow to the largest row count. No Ray actor, distributed
group, or GPU is needed. (The shared session fixture in
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
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from tests.backends.skyrl_train.sample_support_helpers import ragged_support

try:
    from skyrl.backends.skyrl_train.distributed.megatron.rank_microbatch import (
        HostMicrobatch,
        LossTensors,
    )
except ModuleNotFoundError as e:
    if not e.name or (e.name != "megatron" and not e.name.startswith("megatron.")):
        raise
    pytest.skip(f"megatron unavailable: {e}", allow_module_level=True)


def _micro(batch_size: int, seq_len: int = 8, **extra) -> HostMicrobatch:
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    return HostMicrobatch(
        sequences=torch.arange(batch_size * seq_len, dtype=torch.long).reshape(batch_size, seq_len),
        attention_mask=attention_mask,
        position_ids=attention_mask.long().cumsum(-1) - 1,
        num_actions=4,
        **extra,
    )


@pytest.mark.megatron
def test_pads_sub_seq_lengths_to_target_batch_size():
    """``sub_seq_lengths`` is indexed by batch position, so it must grow with the rows.

    ``TokenLayout.build`` raises when ``len(sub_seq_lengths)`` disagrees with the
    batch size, and derives the segment layout from it otherwise.
    """
    sub_seq_lengths = TensorList([torch.tensor([5, 3], dtype=torch.long), torch.tensor([8], dtype=torch.long)])
    micro = _micro(2, sub_seq_lengths=sub_seq_lengths)

    padded = micro.padded(4)

    assert padded.sequences.shape[0] == 4
    assert len(padded.sub_seq_lengths) == 4
    # Real rows are unchanged.
    assert padded.sub_seq_lengths[0].tolist() == [5, 3]
    assert padded.sub_seq_lengths[1].tolist() == [8]
    # Dummy rows describe the single valid token the padded attention_mask carries.
    assert padded.sub_seq_lengths[2].tolist() == [1]
    assert padded.sub_seq_lengths[3].tolist() == [1]
    assert padded.attention_mask[2].tolist() == [1, 0, 0, 0, 0, 0, 0, 0]


@pytest.mark.megatron
def test_padded_sub_seq_lengths_agree_with_attention_mask():
    """Every row's sub-sequence lengths sum to that row's valid-token count, which is
    the invariant ``TokenLayout`` and the packed collator share."""
    attention_mask = torch.zeros((2, 8), dtype=torch.long)
    attention_mask[0, :5] = 1
    attention_mask[1, :3] = 1
    micro = _micro(2, sub_seq_lengths=TensorList([torch.tensor([5], dtype=torch.long), torch.tensor([3])]))
    micro.attention_mask = attention_mask

    padded = micro.padded(5)

    for row in range(5):
        assert int(padded.attention_mask[row].sum()) == int(padded.sub_seq_lengths[row].sum())


@pytest.mark.megatron
def test_pads_multimodal_tensor_lists_with_empty_rows():
    """Dummy rows have no image, so they get the zero-row placeholder the batch builder
    already uses for text-only samples in a mixed batch."""
    pixel_values = TensorList([torch.randn(12, 6), torch.randn(4, 6)])
    image_grid_thw = TensorList([torch.tensor([[1, 3, 4]]), torch.tensor([[1, 2, 2]])])
    micro = _micro(2, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    padded = micro.padded(3)

    assert len(padded.pixel_values) == 3
    assert padded.pixel_values[2].shape == (0, 6)
    assert padded.pixel_values[2].dtype == pixel_values[0].dtype
    assert len(padded.image_grid_thw) == 3
    assert padded.image_grid_thw[2].shape == (0, 3)
    # Empty rows contribute nothing to the concatenated vision input.
    assert torch.cat(padded.pixel_values.tensors).shape == (16, 6)


@pytest.mark.megatron
def test_no_padding_needed_returns_input_unchanged():
    sub_seq_lengths = TensorList([torch.tensor([5], dtype=torch.long), torch.tensor([8], dtype=torch.long)])
    micro = _micro(2, sub_seq_lengths=sub_seq_lengths)

    assert micro.padded(2) is micro
    assert len(micro.sub_seq_lengths) == 2


@pytest.mark.megatron
def test_padding_does_not_mutate_the_input_tensor_list():
    sub_seq_lengths = TensorList([torch.tensor([5], dtype=torch.long)])
    micro = _micro(1, sub_seq_lengths=sub_seq_lengths)

    padded = micro.padded(3)

    assert len(sub_seq_lengths) == 1
    assert len(padded.sub_seq_lengths) == 3


@pytest.mark.megatron
def test_absent_tensor_list_fields_stay_absent():
    micro = _micro(2, sub_seq_lengths=None, pixel_values=None)

    padded = micro.padded(4)

    assert padded.sub_seq_lengths is None
    assert padded.pixel_values is None
    assert padded.num_actions == 4


@pytest.mark.megatron
def test_padding_extends_packed_and_tensor_list_fields():
    routes = PackedTensor.from_segments([torch.ones((8, 2, 2), dtype=torch.int16)])
    support = ragged_support([[[2, 5], [3, 6], [4, 7], [5, 8]]])
    sub_seq_lengths = TensorList([torch.tensor([5, 3])])
    micro = _micro(
        1,
        sub_seq_lengths=sub_seq_lengths,
        rollout_expert_indices=routes,
        rollout_sample_support=support,
        loss=LossTensors(loss_mask=torch.ones((1, 4)), response_mask=torch.ones((1, 4))),
    )

    padded = micro.padded(3)

    assert padded.sub_seq_lengths[0].tolist() == [5, 3]
    assert [row.tolist() for row in padded.sub_seq_lengths[1:]] == [[1], [1]]
    padded_routes = padded.rollout_expert_indices
    assert padded_routes.sequence_lengths.tolist() == [8, 1, 1]
    torch.testing.assert_close(padded_routes.values[:8], routes.values)
    assert padded_routes.values[8:].tolist() == [[[0, 1], [0, 1]], [[0, 1], [0, 1]]]
    padded_support = padded.rollout_sample_support
    assert type(padded_support) is type(support)
    assert padded_support.sequence_lengths.tolist() == [4, 0, 0]
    for padded_column, column in zip(padded_support.columns, support.columns, strict=True):
        torch.testing.assert_close(padded_column, column)
    assert padded.attention_mask[1:].sum(dim=1).tolist() == [1, 1]
    assert not padded.loss.loss_mask[1:].any()
    assert not padded.loss.response_mask[1:].any()
    assert len(sub_seq_lengths) == len(routes) == len(support) == 1


@pytest.mark.megatron
def test_dummy_rows_train_nothing_and_mask_their_routers():
    micro = _micro(
        1,
        router_padding_mask=torch.zeros((1, 8), dtype=torch.bool),
        loss=LossTensors(
            advantages=torch.ones((1, 4)),
            old_action_log_probs=torch.ones((1, 4)),
            real_sample_mask=torch.ones(1, dtype=torch.bool),
            num_microbatches=3,
        ),
    )

    padded = micro.padded(3)

    assert padded.router_padding_mask[1:].all() and not padded.router_padding_mask[0].any()
    assert padded.position_ids[1:].tolist() == [list(range(8))] * 2
    assert padded.loss.real_sample_mask.tolist() == [True, False, False]
    assert not padded.loss.advantages[1:].any() and not padded.loss.old_action_log_probs[1:].any()
    assert padded.loss.num_microbatches == 3 and padded.num_actions == 4
    # The input is not changed.
    assert micro.batch_size == 1 and micro.loss.advantages.shape == (1, 4)


@pytest.mark.megatron
def test_append_tensor_list_padding_zero_count_is_a_no_op():
    field = TensorList([torch.tensor([5], dtype=torch.long)])

    assert append_tensor_list_padding("sub_seq_lengths", field, 0) is field
