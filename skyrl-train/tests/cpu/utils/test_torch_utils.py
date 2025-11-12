from skyrl_train.utils.torch_utils import chunked_cross_entropy_from_log_probs, chunked_entropy_from_logits
import torch
import pytest
import math


def test_chunked_cross_entropy_from_logprobs():
    # Define a small log-probability tensor (batch_size=2, seqlen=3, vocab_size=4)
    logits = [
        [
            [1.0, 2.0, 3.0, 4.0],  # example 1, token 1
            [1.0, 0.0, 0.0, 0.0],  # token 2
            [0.0, 0.0, 0.0, 0.0],
        ],  # token 3 (uniform)
        [
            [0.0, 0.0, 0.0, 0.0],  # example 2, token 1 (uniform)
            [1.0, 2.0, 3.0, 4.0],  # token 2
            [4.0, 3.0, 2.0, 1.0],
        ],  # token 3
    ]
    logits = torch.tensor(logits, dtype=torch.float32)
    logprobs_BSV = torch.log_softmax(logits, dim=-1)  # shape: (2, 3, 4)

    result_BS = chunked_cross_entropy_from_log_probs(logprobs_BSV)

    # For uniform logprobs (all zeros before softmax), entropy should be log(vocab_size) = log(4)
    expected_uniform_entropy = math.log(4.0)  # ≈ 1.386

    assert torch.allclose(result_BS[0, 2], torch.tensor(expected_uniform_entropy), atol=1e-4)
    assert torch.allclose(result_BS[1, 0], torch.tensor(expected_uniform_entropy), atol=1e-4)


def test_chunked_entropy_from_logits():
    # Define a small log-probability tensor (batch_size=2, seqlen=3, vocab_size=4)
    logits = [
        [
            [1.0, 2.0, 3.0, 4.0],  # example 1, token 1
            [1.0, 0.0, 0.0, 0.0],  # token 2
            [0.0, 0.0, 0.0, 0.0],
        ],  # token 3 (uniform)
        [
            [0.0, 0.0, 0.0, 0.0],  # example 2, token 1 (uniform)
            [1.0, 2.0, 3.0, 4.0],  # token 2
            [4.0, 3.0, 2.0, 1.0],
        ],  # token 3
    ]
    logits = torch.tensor(logits, dtype=torch.float32)

    result_BS = chunked_entropy_from_logits(logits)

    # For uniform logprobs (all zeros before softmax), entropy should be log(vocab_size) = log(4)
    expected_uniform_entropy = math.log(4.0)  # ≈ 1.386

    assert torch.allclose(result_BS[0, 2], torch.tensor(expected_uniform_entropy), atol=1e-4)
    assert torch.allclose(result_BS[1, 0], torch.tensor(expected_uniform_entropy), atol=1e-4)


def test_chunked_entropy_from_logits_with_attention_mask():
    """Test that attention mask correctly zeros out entropy for padded positions."""
    # Define logits (batch_size=2, seqlen=4, vocab_size=4)
    logits = torch.randn(2, 4, 4, dtype=torch.float32)

    # Create attention mask: 1 for valid tokens, 0 for padding
    # First sequence: all valid
    # Second sequence: first 2 valid, last 2 padded
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1],  # all valid
            [1, 1, 0, 0],  # last 2 are padding
        ],
        dtype=torch.float32,
    )

    # Calculate entropy with attention mask
    result_with_mask = chunked_entropy_from_logits(logits, attention_mask=attention_mask)

    # Verify that padded positions have zero entropy
    assert result_with_mask[1, 2] == 0.0, "Padded position should have zero entropy"
    assert result_with_mask[1, 3] == 0.0, "Padded position should have zero entropy"

    # Verify that valid positions have non-zero entropy
    assert result_with_mask[0, 0] > 0.0, "Valid position should have non-zero entropy"
    assert result_with_mask[1, 0] > 0.0, "Valid position should have non-zero entropy"

    # Calculate entropy without mask for the valid positions
    result_without_mask = chunked_entropy_from_logits(logits)

    # Entropy values for valid positions should be the same with or without mask
    assert torch.allclose(result_with_mask[0, :], result_without_mask[0, :], atol=1e-6)
    assert torch.allclose(result_with_mask[1, :2], result_without_mask[1, :2], atol=1e-6)


def test_chunked_entropy_from_logits_attention_mask_shape_validation():
    """Test that attention mask shape validation works correctly."""
    logits = torch.randn(2, 3, 4, dtype=torch.float32)

    # Wrong shape: different batch size
    wrong_mask_batch = torch.ones(3, 3, dtype=torch.float32)

    # For wrong batch size
    with pytest.raises(ValueError, match="does not match logits shape"):
        chunked_entropy_from_logits(logits, attention_mask=wrong_mask_batch)

    # Wrong shape: different sequence length
    wrong_mask_seqlen = torch.ones(2, 4, dtype=torch.float32)

    # For wrong sequence length
    with pytest.raises(ValueError, match="does not match logits shape"):
        chunked_entropy_from_logits(logits, attention_mask=wrong_mask_seqlen)

    # Correct shape should work
    correct_mask = torch.ones(2, 3, dtype=torch.float32)
    result = chunked_entropy_from_logits(logits, attention_mask=correct_mask)
    assert result.shape == (2, 3)
