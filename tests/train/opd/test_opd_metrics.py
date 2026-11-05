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
    TopKStats,
    informativeness,
    is_repetitive,
    rollout_correct,
    topk_stats,
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


# ---------------------------------------------------------------------------
# Top-K diagnostics
# ---------------------------------------------------------------------------


def test_topk_stats_add_and_report_only_once_measured():
    assert TopKStats().metrics() == {}
    stats = TopKStats(positions=4, overlap=2.0, advantage=-0.3, advantage_positions=3)
    total = stats + stats
    assert total.positions == 8 and total.advantage_positions == 6
    assert total.metrics() == {
        "opd/overlap_ratio": 0.5,
        "opd/overlap_positions": 8,
        "opd/overlap_token_advantage": pytest.approx(-0.1),
    }


def _log_distribution(rng, size):
    logits = rng.normal(size=size)
    return logits - np.log(np.exp(logits).sum())


def test_identical_top_k_overlap_fully_with_zero_advantage():
    ids = np.array([[3, 1, 2]] * 2)
    logprobs = np.log(np.array([[0.5, 0.3, 0.1]] * 2))
    metrics = topk_stats(ids, logprobs, ids, logprobs, np.ones(2, bool)).metrics()

    assert metrics["opd/overlap_ratio"] == 1.0
    assert metrics["opd/overlap_token_advantage"] == pytest.approx(0.0, abs=1e-12)


def test_disjoint_top_k_have_no_overlap_and_no_advantage():
    logprobs = np.log(np.array([[0.6, 0.4]]))
    metrics = topk_stats(np.array([[1, 2]]), logprobs, np.array([[3, 4]]), logprobs, np.ones(1, bool)).metrics()

    assert metrics["opd/overlap_ratio"] == 0.0
    assert "opd/overlap_token_advantage" not in metrics


def _loop_reference(s_ids, s_lp, t_ids, t_lp, mask):
    """Eqs. 6 and 7, position by position from their definitions."""
    k = s_ids.shape[1]
    overlaps, advantages = [], []
    for t in np.flatnonzero(mask):
        student = {int(v): float(lp) for v, lp in zip(s_ids[t], s_lp[t]) if v >= 0}
        teacher = {int(v): float(lp) for v, lp in zip(t_ids[t], t_lp[t])}
        shared = [v for v in student if v in teacher]
        overlaps.append(len(shared) / k)
        if shared:
            z_p = sum(np.exp(student[v]) for v in shared)
            z_q = sum(np.exp(teacher[v]) for v in shared)
            advantages.append(
                sum(
                    np.exp(student[v]) / z_p * (np.log(np.exp(teacher[v]) / z_q) - np.log(np.exp(student[v]) / z_p))
                    for v in shared
                )
                / len(shared)
            )
    return overlaps, advantages


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_topk_stats_match_a_loop_over_the_definitions(seed):
    rng = np.random.default_rng(seed)
    tokens, vocab, k = 40, 24, 8
    s_ids, s_lp, t_ids, t_lp = [], [], [], []
    for _ in range(tokens):
        p, q = _log_distribution(rng, vocab), _log_distribution(rng, vocab)
        top_p, top_q = np.argsort(-p)[:k], np.argsort(-q)[:k]
        s_ids.append(top_p), s_lp.append(p[top_p]), t_ids.append(top_q), t_lp.append(q[top_q])
    s_ids, s_lp, t_ids, t_lp = map(np.array, (s_ids, s_lp, t_ids, t_lp))
    # Short student rows (-1 and -inf past their end), as capture pads them, and masked positions.
    s_ids[:5, 6:], s_lp[:5, 6:] = -1, -np.inf
    mask = rng.random(tokens) > 0.2

    metrics = topk_stats(s_ids, s_lp, t_ids, t_lp, mask).metrics()
    overlaps, advantages = _loop_reference(s_ids, s_lp, t_ids, t_lp, mask)

    assert metrics["opd/overlap_positions"] == len(overlaps) == mask.sum()
    assert metrics["opd/overlap_ratio"] == pytest.approx(np.mean(overlaps))
    assert metrics["opd/overlap_token_advantage"] == pytest.approx(np.mean(advantages))
    # Every position's Eq. 7 value is a negative KL.
    assert all(value <= 1e-12 for value in advantages)
