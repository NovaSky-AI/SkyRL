"""Tests for the cuDNN DSA backward top-k alignment.

``align_topk_axis`` and ``_check_backward_contract`` are the parts of
``skyrl.backends.skyrl_train.patches.megatron.patch_dsa_backward_topk_align`` that carry
the logic; neither needs Megatron or a GPU, so they are tested in the CPU lane. The
kernel-level check against dense autograd needs SM90+ and FlashMLA and lives in
``tests/backends/skyrl_train/gpu/gpu_ci/patches/megatron/test_dsa_backward_topk_align_cuda.py``.

Run with:
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/patches/megatron/test_dsa_backward_topk_align.py
"""

import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron.patch_dsa_backward_topk_align import (
    _BLOCK_TILE,
    _check_backward_contract,
    align_topk_axis,
)

# Marked like the rest of this folder, so the CPU megatron job runs it.
pytestmark = pytest.mark.megatron


@pytest.mark.parametrize(
    "width, expected",
    [
        (2051, 2112),  # index_topk=2048 + pool_size-1=3, the kpool indexer's width
        (1, 64),
        (63, 64),
        (65, 128),
        (2560, 2560),  # a 512-aligned width is already 64-aligned
        (64, 64),
    ],
)
def test_width_is_rounded_up_to_the_block_tile(width, expected):
    idxs = torch.randint(0, 100, (2, 4, width), dtype=torch.int32)

    aligned = align_topk_axis(idxs)

    assert aligned.shape[-1] == expected
    assert expected % _BLOCK_TILE == 0


def test_already_aligned_input_is_returned_unchanged():
    """No copy when there is nothing to pad."""
    idxs = torch.randint(0, 100, (2, 4, 2560), dtype=torch.int32)

    assert align_topk_axis(idxs) is idxs


def test_padding_preserves_the_real_columns_and_pads_with_zero():
    """The pad must be 0: the kernel dereferences mKV[topk_idx] with no negativity check."""
    idxs = torch.randint(1, 100, (2, 3, 100), dtype=torch.int32)

    aligned = align_topk_axis(idxs)

    assert torch.equal(aligned[..., :100], idxs)
    assert torch.all(aligned[..., 100:] == 0)
    assert aligned.dtype == torch.int32


def test_contract_check_passes_on_valid_inputs():
    skv, b, width = 128, 2, 64
    idxs = torch.randint(0, skv * b, (4, width), dtype=torch.int32)
    lengths = torch.full((4,), width, dtype=torch.int32)

    _check_backward_contract(idxs, lengths, skv, b, width)


def test_contract_check_rejects_a_negative_index():
    skv, b, width = 128, 2, 64
    idxs = torch.zeros((4, width), dtype=torch.int32)
    idxs[1, 5] = -1
    lengths = torch.full((4,), width, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="negative_index=True"):
        _check_backward_contract(idxs, lengths, skv, b, width)


def test_contract_check_rejects_an_index_past_the_kv_extent():
    skv, b, width = 128, 2, 64
    idxs = torch.zeros((4, width), dtype=torch.int32)
    idxs[0, 0] = skv * b
    lengths = torch.full((4,), width, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="index_ge_skv_times_b=True"):
        _check_backward_contract(idxs, lengths, skv, b, width)


def test_contract_check_rejects_a_topk_length_past_the_width():
    skv, b, width = 128, 2, 64
    idxs = torch.zeros((4, width), dtype=torch.int32)
    lengths = torch.full((4,), width + 1, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="topk_length_gt_width=True"):
        _check_backward_contract(idxs, lengths, skv, b, width)


def test_contract_check_rejects_a_non_int32_dtype():
    """cuDNN reinterprets both tensors as int32 without checking."""
    skv, b, width = 128, 2, 64
    idxs = torch.zeros((4, width), dtype=torch.int64)
    lengths = torch.full((4,), width, dtype=torch.int32)

    with pytest.raises(RuntimeError, match="reinterprets these as int32"):
        _check_backward_contract(idxs, lengths, skv, b, width)


def test_contract_check_rejects_a_non_contiguous_index_tensor():
    """The sm100 path assumes packed rows but takes strides from the tensors."""
    skv, b, width = 128, 2, 64
    idxs = torch.zeros((4, width * 2), dtype=torch.int32)[:, ::2]
    lengths = torch.full((4,), width, dtype=torch.int32)
    assert not idxs.is_contiguous()

    with pytest.raises(RuntimeError, match="assumes packed rows"):
        _check_backward_contract(idxs, lengths, skv, b, width)


def test_contract_check_error_reports_the_observed_ranges():
    """The message must carry the numbers, since the alternative is an async CUDA fault."""
    skv, b, width = 128, 2, 64
    idxs = torch.zeros((4, width), dtype=torch.int32)
    idxs[2, 3] = -7
    lengths = torch.full((4,), width, dtype=torch.int32)

    with pytest.raises(RuntimeError) as excinfo:
        _check_backward_contract(idxs, lengths, skv, b, width)

    message = str(excinfo.value)
    assert "global_idxs=[-7, 0]" in message
    assert f"width={width} skv={skv} b={b}" in message


def test_contract_check_error_survives_an_empty_index_tensor():
    """A zero-width index tensor must still produce the contract error, not a secondary
    ``min()``-of-empty failure while formatting it."""
    idxs = torch.zeros((4, 0), dtype=torch.int32)
    lengths = torch.ones((4,), dtype=torch.int32)

    with pytest.raises(RuntimeError, match="topk_length_gt_width=True") as excinfo:
        _check_backward_contract(idxs, lengths, 128, 2, 0)

    assert "global_idxs=empty topk_length=[1, 1]" in str(excinfo.value)
