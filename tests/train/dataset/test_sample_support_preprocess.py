import pytest
import torch

from skyrl.train.dataset.preprocess import build_dense_sample_support


def test_sample_support_is_response_aligned_int32():
    support = build_dense_sample_support(
        [[[10, 11, -1], [12, -1, -1]], [[20, 21, 22]]],
        [[10, 12], [20]],
        [[1, 1], [1]],
        sequence_length=5,
        top_k=3,
        eos_token_id=2,
    )

    assert support is not None
    assert support.dtype == torch.int32
    assert support[0].tolist() == [[-1, -1, -1]] * 3 + [[10, 11, -1], [12, -1, -1]]
    assert support[1].tolist() == [[-1, -1, -1]] * 4 + [[20, 21, 22]]


def test_empty_loss_masked_rows_and_loss_bearing_synthetic_eos_are_preserved():
    support = build_dense_sample_support(
        [[[7, 8], [], []]],
        [[7, 9, 2]],
        [[1, 0, 1]],
        sequence_length=4,
        top_k=2,
        eos_token_id=2,
    )

    assert support.tolist() == [[[-1, -1], [7, 8], [-1, -1], [-1, -1]]]


def test_empty_loss_bearing_non_eos_is_rejected():
    with pytest.raises(ValueError, match="loss-bearing non-EOS"):
        build_dense_sample_support(
            [[[]]],
            [[3]],
            [[1]],
            sequence_length=2,
            top_k=2,
            eos_token_id=2,
        )


def test_multiple_loss_bearing_unsupported_eos_are_rejected():
    with pytest.raises(ValueError, match="more than one loss-bearing unsupported token"):
        build_dense_sample_support(
            [[[], []]],
            [[2, 2]],
            [[1, 1]],
            sequence_length=2,
            top_k=2,
            eos_token_id=2,
        )


def test_loss_bearing_sampled_token_must_be_in_support():
    with pytest.raises(ValueError, match="sampled token 3 is missing"):
        build_dense_sample_support(
            [[[2, 4]]],
            [[3]],
            [[1]],
            sequence_length=2,
            top_k=2,
            eos_token_id=2,
        )
