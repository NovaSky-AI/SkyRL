"""CPU tests for the OPD rollout diagnostics (skyrl/train/opd/metrics.py).

Run with:
uv run --isolated --extra dev pytest tests/train/opd/test_opd_metrics.py
"""

import random

import numpy as np
import pytest
import torch

from skyrl.train.opd.metrics import (
    REPETITION_TAIL_TOKENS,
    informativeness,
    is_repetitive,
    rollout_correct,
    truncation_rate,
)


def test_truncation_rate_counts_length_stops():
    assert truncation_rate(["stop", "length", "length", "abort"]) == 0.5
    assert truncation_rate(["stop"]) == 0.0
    assert truncation_rate([]) == 0.0


def test_is_repetitive_needs_a_long_compressible_tail():
    rng = random.Random(0)
    varied = [rng.randrange(150_000) for _ in range(2 * REPETITION_TAIL_TOKENS)]

    assert not is_repetitive([7, 8] * (REPETITION_TAIL_TOKENS // 2 - 1))  # two tokens short of eligible
    assert is_repetitive([7, 8] * (REPETITION_TAIL_TOKENS // 2))
    assert not is_repetitive(varied)
    # Only the tail counts: a varied rollout that ends in a loop is repetitive.
    assert is_repetitive(varied + [11, 12, 13, 14] * 1_000)


@pytest.mark.parametrize(
    "reward, correct",
    [
        (1.0, True),
        (0.0, False),
        (-1.0, False),
        (2, True),
        ([0.0, 0.0, 1.0], True),
        ([1.0, 0.0, 0.0], False),  # only the last token's reward counts, as in pass@n
        ([], False),
        (torch.tensor([0.0, 0.5]), True),
        (np.array([0.0, -0.5]), False),
    ],
)
def test_rollout_correct_follows_pass_at_n(reward, correct):
    assert rollout_correct(reward) is correct


def test_informativeness_is_the_gap_between_correct_and_incorrect_rollouts():
    log_ratio = torch.tensor([[0.7, 0.6, 0.5], [9.0, 1.4, 1.3], [0.2, 0.4, 0.6]])
    loss_mask = torch.tensor([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])

    value = informativeness(log_ratio, loss_mask, torch.tensor([True, False, True]))

    # Correct rows average 0.6 and 0.4; the incorrect row averages 1.35 over its masked tokens only.
    assert value == pytest.approx((0.6 + 0.4) / 2 - 1.35)


def test_informativeness_leaves_out_rows_with_no_trained_tokens():
    log_ratio = torch.tensor([[1.0, 1.0], [-1.0, -1.0], [5.0, 5.0], [100.0, 100.0]])
    loss_mask = torch.tensor([[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])

    # Row 2 trains nothing (overlong filtering) and row 3 is batch padding: neither counts, whatever its flag.
    assert informativeness(log_ratio, loss_mask, torch.tensor([True, False, True, False])) == pytest.approx(2.0)


def test_informativeness_needs_both_groups():
    log_ratio, loss_mask = torch.ones(2, 3), torch.ones(2, 3)

    assert informativeness(log_ratio, loss_mask, torch.tensor([True, True])) is None
    assert informativeness(log_ratio, loss_mask, torch.tensor([False, False])) is None
