"""Unit tests for to_te_attention_mask.

Run with:
  uv run --extra dev -- pytest tests/backends/skyrl_train/distributed/test_mask_utils.py
"""

import torch

from skyrl.backends.skyrl_train.distributed.megatron.mask_utils import (
    to_te_attention_mask,
)


def test_expands_2d_keep_mask_to_4d_padding_mask():
    """A 2-D keep-mask becomes a [batch, 1, 1, seq] boolean padding mask where True marks padding."""
    keep_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    out = to_te_attention_mask(keep_mask)
    assert out.shape == (2, 1, 1, 3)
    assert out.dtype == torch.bool
    assert out[0, 0, 0].tolist() == [False, False, True]
    assert out[1, 0, 0].tolist() == [False, False, False]


def test_preserves_distinct_rows_for_micro_batch_gt_one():
    """Each batch row keeps its own padding pattern, which is the micro_batch_size > 1 case."""
    keep_mask = torch.tensor([[1, 0, 0], [0, 1, 1], [1, 1, 0]])
    out = to_te_attention_mask(keep_mask)
    assert out.shape == (3, 1, 1, 3)
    assert torch.equal(out, (~keep_mask.bool())[:, None, None, :])


def test_accepts_boolean_mask():
    """A boolean keep-mask is handled the same as an integer keep-mask."""
    keep_mask = torch.tensor([[True, True, False]])
    out = to_te_attention_mask(keep_mask)
    assert out.tolist() == [[[[False, False, True]]]]


def test_returns_none_unchanged():
    """Packed sequences pass None, which must be forwarded unchanged."""
    assert to_te_attention_mask(None) is None


def test_returns_higher_rank_mask_unchanged():
    """An already-expanded mask is returned without modification."""
    mask = torch.zeros(2, 1, 1, 5, dtype=torch.bool)
    assert to_te_attention_mask(mask) is mask
