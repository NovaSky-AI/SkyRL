"""
Tests for the worker-side advantage-split log-ratio metric (multi-epoch direction check).

uv run --isolated --extra dev pytest tests/train/algorithms/test_advantage_split_logratio.py
"""

import pytest
import torch

from skyrl.backends.skyrl_train.workers.worker_utils import (
    ADVANTAGE_SPLIT_LOGRATIO_KEYS,
    compute_advantage_split_logratio_metrics,
)

POS_KEY, NEG_KEY = ADVANTAGE_SPLIT_LOGRATIO_KEYS


def test_splits_by_advantage_sign_and_respects_mask():
    old = torch.zeros(2, 3)
    new = torch.tensor([[0.3, 0.1, 9.0], [-0.2, -0.4, 9.0]])
    adv = torch.tensor([[1.0, 1.0, 1.0], [-2.0, -2.0, -2.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])  # last token masked out

    m = compute_advantage_split_logratio_metrics(new, old, adv, mask)

    assert m[POS_KEY] == pytest.approx(0.2)
    assert m[NEG_KEY] == pytest.approx(-0.3)


def test_zero_advantage_tokens_are_excluded():
    old = torch.zeros(1, 3)
    new = torch.tensor([[0.5, 7.0, -0.5]])
    adv = torch.tensor([[1.0, 0.0, -1.0]])

    m = compute_advantage_split_logratio_metrics(new, old, adv, None)

    assert m[POS_KEY] == pytest.approx(0.5)
    assert m[NEG_KEY] == pytest.approx(-0.5)


def test_no_tokens_of_a_sign_reads_zero():
    old = torch.zeros(1, 2)
    new = torch.tensor([[0.4, 0.2]])
    adv = torch.ones(1, 2)

    m = compute_advantage_split_logratio_metrics(new, old, adv, torch.ones(1, 2))

    assert set(m) == {POS_KEY, NEG_KEY}
    assert m[POS_KEY] == pytest.approx(0.3)
    assert m[NEG_KEY] == 0.0
