"""
Tests that loss_fn_outputs extraction correctly handles right-aligned (left-padded)
response tensors.

Regression test for https://github.com/NovaSky-AI/SkyRL/issues/1304

The response-level tensors (action_log_probs, elementwise_loss, action_mask) are
right-aligned in the batch — padding is on the left. The extraction code must use
[-valid_len:] (right-aligned slicing) instead of [:valid_len] (left-aligned slicing)
to return the actual response values rather than padding/prompt values.

uv run --isolated --extra dev pytest tests/backends/skyrl_train/workers/test_loss_fn_outputs_slicing.py
"""

import pytest
import torch


def _extract_loss_fn_outputs_sft(
    action_log_probs: torch.Tensor,
    elementwise_loss: torch.Tensor,
    action_mask: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
) -> list[dict]:
    """
    Reproduces the SFT loss_fn_outputs extraction logic from both
    worker.py and megatron_model_wrapper.py.
    """
    batch_size = action_log_probs.shape[0]
    loss_fn_outputs = []
    for i in range(batch_size):
        if action_mask is not None:
            valid_len = int(action_mask[i].sum().item())
        elif loss_mask is not None:
            valid_len = int(loss_mask[i].sum().item())
        else:
            valid_len = action_log_probs.shape[1]

        loss_fn_outputs.append(
            {
                "logprobs": action_log_probs[i, -valid_len:].detach().cpu().tolist() if valid_len > 0 else [],
                "elementwise_loss": elementwise_loss[i, -valid_len:].detach().cpu().tolist() if valid_len > 0 else [],
            }
        )
    return loss_fn_outputs


def _extract_loss_fn_outputs_rl(
    action_log_probs: torch.Tensor,
    action_mask: torch.Tensor | None,
    loss_mask: torch.Tensor | None,
) -> list[dict]:
    """
    Reproduces the RL loss_fn_outputs extraction logic from both
    worker.py and megatron_model_wrapper.py.
    """
    batch_size = action_log_probs.shape[0]
    seq_len = action_log_probs.shape[1]

    if action_mask is not None:
        valid_lens = action_mask.sum(dim=1).int().tolist()
    elif loss_mask is not None:
        valid_lens = loss_mask.sum(dim=1).int().tolist()
    else:
        valid_lens = [seq_len] * batch_size

    detached_log_probs = action_log_probs.detach().cpu()
    loss_fn_outputs = []
    for i, valid_len in enumerate(valid_lens):
        loss_fn_outputs.append(
            {
                "logprobs": detached_log_probs[i, -valid_len:].tolist() if valid_len > 0 else [],
            }
        )
    return loss_fn_outputs


class TestLossFnOutputsSlicing:
    """Verify that loss_fn_outputs extraction uses right-aligned slicing on
    right-aligned (left-padded) response tensors."""

    def test_sft_path_right_aligned_with_padding(self):
        """SFT path: with left-padding, [-valid_len:] must return the real response values."""
        # Batch of 2 sequences, max_response_len=5
        # Sequence 0: 3 real response tokens (2 padding tokens on the left)
        # Sequence 1: 5 real response tokens (no padding)
        action_log_probs = torch.tensor(
            [
                [0.0, 0.0, -1.1, -2.2, -3.3],  # pad, pad, real, real, real
                [-0.5, -1.5, -2.5, -3.5, -4.5],  # all real
            ]
        )
        elementwise_loss = torch.tensor(
            [
                [0.0, 0.0, 1.1, 2.2, 3.3],
                [0.5, 1.5, 2.5, 3.5, 4.5],
            ]
        )
        action_mask = torch.tensor(
            [
                [0, 0, 1, 1, 1],  # right-aligned: padding on left
                [1, 1, 1, 1, 1],
            ]
        )

        outputs = _extract_loss_fn_outputs_sft(action_log_probs, elementwise_loss, action_mask, None)

        # Sequence 0: should get the 3 rightmost values (the real response)
        assert outputs[0]["logprobs"] == pytest.approx([-1.1, -2.2, -3.3])
        assert outputs[0]["elementwise_loss"] == pytest.approx([1.1, 2.2, 3.3])

        # Sequence 1: full length, no difference
        assert outputs[1]["logprobs"] == pytest.approx([-0.5, -1.5, -2.5, -3.5, -4.5])
        assert outputs[1]["elementwise_loss"] == pytest.approx([0.5, 1.5, 2.5, 3.5, 4.5])

    def test_sft_path_left_slicing_would_return_padding(self):
        """Demonstrates the bug: [:valid_len] returns padding, not real values."""
        action_log_probs = torch.tensor(
            [
                [0.0, 0.0, -1.1, -2.2, -3.3],  # pad, pad, real, real, real
            ]
        )
        elementwise_loss = torch.tensor(
            [
                [0.0, 0.0, 1.1, 2.2, 3.3],
            ]
        )
        action_mask = torch.tensor([[0, 0, 1, 1, 1]])
        valid_len = int(action_mask[0].sum().item())  # 3

        # The WRONG way (old code): left-aligned slice
        wrong_logprobs = action_log_probs[0, :valid_len].tolist()
        # The CORRECT way (fixed code): right-aligned slice
        correct_logprobs = action_log_probs[0, -valid_len:].tolist()

        # Wrong extracts padding values [0.0, 0.0, -1.1] instead of real [-1.1, -2.2, -3.3]
        assert wrong_logprobs == pytest.approx([0.0, 0.0, -1.1])
        assert correct_logprobs == pytest.approx([-1.1, -2.2, -3.3])
        assert wrong_logprobs != correct_logprobs

    def test_rl_path_right_aligned_with_padding(self):
        """RL path: with left-padding, [-valid_len:] must return the real response values."""
        action_log_probs = torch.tensor(
            [
                [0.0, 0.0, -1.0, -2.0, -3.0],  # 2 pad + 3 real
                [0.0, -0.1, -0.2, -0.3, -0.4],  # 1 pad + 4 real
            ]
        )
        action_mask = torch.tensor(
            [
                [0, 0, 1, 1, 1],
                [0, 1, 1, 1, 1],
            ]
        )

        outputs = _extract_loss_fn_outputs_rl(action_log_probs, action_mask, None)

        assert outputs[0]["logprobs"] == pytest.approx([-1.0, -2.0, -3.0])
        assert outputs[1]["logprobs"] == pytest.approx([-0.1, -0.2, -0.3, -0.4])

    def test_rl_path_with_loss_mask_fallback(self):
        """RL path: when action_mask is None, falls back to loss_mask."""
        action_log_probs = torch.tensor(
            [
                [0.0, 0.0, -1.0, -2.0, -3.0],
            ]
        )
        loss_mask = torch.tensor(
            [
                [0.0, 0.0, 1.0, 1.0, 1.0],
            ]
        )

        outputs = _extract_loss_fn_outputs_rl(action_log_probs, None, loss_mask)
        assert outputs[0]["logprobs"] == pytest.approx([-1.0, -2.0, -3.0])

    def test_no_padding_same_result(self):
        """When there is no padding (valid_len == seq_len), both slicings agree."""
        action_log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        action_mask = torch.tensor([[1, 1, 1]])

        outputs = _extract_loss_fn_outputs_rl(action_log_probs, action_mask, None)
        assert outputs[0]["logprobs"] == pytest.approx([-1.0, -2.0, -3.0])

    def test_single_valid_token(self):
        """Edge case: only 1 valid token at the rightmost position."""
        action_log_probs = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, -5.0],
            ]
        )
        elementwise_loss = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 5.0],
            ]
        )
        action_mask = torch.tensor([[0, 0, 0, 0, 1]])

        outputs = _extract_loss_fn_outputs_sft(action_log_probs, elementwise_loss, action_mask, None)
        assert outputs[0]["logprobs"] == pytest.approx([-5.0])
        assert outputs[0]["elementwise_loss"] == pytest.approx([5.0])

    def test_zero_valid_len_returns_empty(self):
        """Edge case: valid_len=0 (fully padded sequence) must return empty lists, not the full tensor.

        In Python, -0 == 0, so tensor[-0:] == tensor[0:] which returns the entire
        tensor. The guard `if valid_len > 0 else []` prevents this.
        """
        action_log_probs = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],  # fully padded
            ]
        )
        elementwise_loss = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ]
        )
        action_mask = torch.tensor([[0, 0, 0, 0, 0]])  # no valid tokens

        # SFT path
        sft_outputs = _extract_loss_fn_outputs_sft(action_log_probs, elementwise_loss, action_mask, None)
        assert sft_outputs[0]["logprobs"] == []
        assert sft_outputs[0]["elementwise_loss"] == []

        # RL path
        rl_outputs = _extract_loss_fn_outputs_rl(action_log_probs, action_mask, None)
        assert rl_outputs[0]["logprobs"] == []

    def test_no_mask_returns_full_sequence(self):
        """When both masks are None, return the full sequence."""
        action_log_probs = torch.tensor([[-1.0, -2.0, -3.0]])
        elementwise_loss = torch.tensor([[1.0, 2.0, 3.0]])

        outputs = _extract_loss_fn_outputs_sft(action_log_probs, elementwise_loss, None, None)
        assert outputs[0]["logprobs"] == pytest.approx([-1.0, -2.0, -3.0])
        assert outputs[0]["elementwise_loss"] == pytest.approx([1.0, 2.0, 3.0])
