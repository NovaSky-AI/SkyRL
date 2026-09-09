"""Row-id derivation for packed sampler support."""

from typing import List, Tuple

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.megatron.token_metadata import (
    TokenMetadataLayout,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.packed_tensor import (
    PackedTensor,
    cu_seqlens_from_lengths,
)
from skyrl.backends.skyrl_train.utils.sample_support import (
    SAMPLE_SUPPORT_FIELD,
    SAMPLE_SUPPORT_NO_ROW,
    SAMPLE_SUPPORT_TORCH_DTYPE,
    align_sample_support_row_ids,
)

TOP_K = 3
# Anti-correlated lengths exercise different support offsets.
LENGTHS: List[Tuple[int, int]] = [(2, 3), (4, 2)]


def _attention_mask(lengths: List[Tuple[int, int]]) -> torch.Tensor:
    """Left-padded mask, as ``convert_prompts_responses_to_batch_tensors`` builds it."""
    totals = [prompt + response for prompt, response in lengths]
    mask = torch.zeros((len(totals), max(totals)), dtype=torch.bool)
    for row, total in enumerate(totals):
        mask[row, max(totals) - total :] = True
    return mask


def _layout(lengths: List[Tuple[int, int]], *, packed: bool = False, align: int = 1) -> TokenMetadataLayout:
    mask = _attention_mask(lengths)
    totals = [prompt + response for prompt, response in lengths]
    if not packed:
        return TokenMetadataLayout(
            attention_mask=mask,
            sequence_lengths=totals,
            aligned_sequence_length=mask.shape[1],
        )
    padded = [total + (-total % align) for total in totals]
    return TokenMetadataLayout(
        attention_mask=mask,
        sequence_lengths=totals,
        aligned_sequence_length=sum(padded),
        padded_sequence_lengths=padded,
        cu_seqlens_padded=torch.tensor([0, *torch.tensor(padded).cumsum(0).tolist()], dtype=torch.int32),
    )


def _support(lengths: List[Tuple[int, int]]) -> PackedTensor:
    """Distinct support rows, so a gather by id pins the exact row it landed on."""
    response_lens = [response for _, response in lengths]
    total_rows = sum(response_lens)
    values = torch.arange(total_rows * TOP_K, dtype=SAMPLE_SUPPORT_TORCH_DTYPE).reshape(total_rows, TOP_K)
    return PackedTensor(values, cu_seqlens_from_lengths(response_lens))


def test_row_ids_land_on_the_positions_that_predict_response_tokens():
    """Support includes the last prompt position and excludes the last response position."""
    row_ids = align_sample_support_row_ids(_support(LENGTHS), _layout(LENGTHS))

    assert row_ids.dtype == torch.int64
    # Trajectory 0 is p=2, r=3, so ids 0..2 sit at real positions 1..3; trajectory 1 is p=4, r=2.
    assert row_ids.tolist() == [
        [SAMPLE_SUPPORT_NO_ROW, 0, 1, 2, SAMPLE_SUPPORT_NO_ROW, SAMPLE_SUPPORT_NO_ROW],
        [SAMPLE_SUPPORT_NO_ROW, SAMPLE_SUPPORT_NO_ROW, SAMPLE_SUPPORT_NO_ROW, 3, 4, SAMPLE_SUPPORT_NO_ROW],
    ]


def test_row_ids_gather_the_support_rows_of_each_response_token():
    support = _support(LENGTHS)

    row_ids = align_sample_support_row_ids(support, _layout(LENGTHS))

    scored = row_ids >= 0
    gathered = support.values[row_ids[scored]]
    assert torch.equal(gathered, support.values)
    assert scored.sum(dim=1).tolist() == [response for _, response in LENGTHS]


@pytest.mark.parametrize("align", [1, 4])
def test_row_ids_follow_megatron_packed_padding(align):
    """Sequence packing interleaves each sequence with its alignment padding."""
    row_ids = align_sample_support_row_ids(_support(LENGTHS), _layout(LENGTHS, packed=True, align=align))

    padded_lengths = [total + (-total % align) for total in (5, 6)]
    assert row_ids.shape == (1, sum(padded_lengths))
    packed = row_ids.squeeze(0).tolist()
    assert packed[1:4] == [0, 1, 2]
    assert packed[padded_lengths[0] + 3 : padded_lengths[0] + 5] == [3, 4]


@pytest.mark.parametrize("selection", ["chunk", "slice"])
def test_row_ids_rebase_under_batch_selection(selection):
    support = _support(LENGTHS)
    batch = TrainingInputBatch({"attention_mask": _attention_mask(LENGTHS).long(), SAMPLE_SUPPORT_FIELD: support})
    selected = batch.chunk(1)[1] if selection == "chunk" else batch.slice(1, 2)

    row_ids = align_sample_support_row_ids(selected[SAMPLE_SUPPORT_FIELD], _layout(LENGTHS[1:]))

    assert row_ids[row_ids >= 0].tolist() == [0, 1]
    assert torch.equal(selected[SAMPLE_SUPPORT_FIELD].values[row_ids[row_ids >= 0]], support.segment(1))


def test_row_ids_accept_an_empty_padding_segment():
    """A synthetic batch row attends one token and generates nothing, so it scores no position."""
    support = PackedTensor(torch.empty((0, TOP_K), dtype=SAMPLE_SUPPORT_TORCH_DTYPE), cu_seqlens_from_lengths([0]))
    mask = torch.zeros((1, 4), dtype=torch.bool)
    mask[0, 0] = True
    layout = TokenMetadataLayout(attention_mask=mask, sequence_lengths=[1], aligned_sequence_length=4)

    row_ids = align_sample_support_row_ids(support, layout)

    assert torch.all(row_ids == SAMPLE_SUPPORT_NO_ROW)


def test_row_ids_reject_a_trajectory_with_no_prompt_token():
    """With no prompt there is no position whose logit predicts the first response token."""
    support = PackedTensor.from_segments([torch.zeros((3, TOP_K), dtype=SAMPLE_SUPPORT_TORCH_DTYPE)])
    layout = TokenMetadataLayout(
        attention_mask=torch.ones((1, 3), dtype=torch.bool),
        sequence_lengths=[3],
        aligned_sequence_length=3,
    )

    with pytest.raises(ValueError, match="predicts its first response token"):
        align_sample_support_row_ids(support, layout)


def test_row_ids_reject_a_segment_count_mismatch():
    with pytest.raises(ValueError, match="segments for"):
        align_sample_support_row_ids(_support(LENGTHS), _layout(LENGTHS[1:]))
