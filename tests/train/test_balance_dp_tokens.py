"""DP token balancing of the training step (trainer.balance_dp_tokens)."""

from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.dataset.preprocess import balanced_dp_order
from skyrl.train.trainer import RayPPOTrainer


def _slice_sums(lengths, order, dp):
    per = len(order) // dp
    return [sum(lengths[i] for i in order[r * per : (r + 1) * per]) for r in range(dp)]


class TestBalancedDpOrder:
    @pytest.mark.parametrize(
        "lengths,dp",
        [
            ([4000] * 8 + [300] * 24, 4),
            ([4300, 4200, 4100, 900, 800, 700, 600, 100], 2),
            (list(range(1, 257)), 32),
        ],
        ids=["long-group-first", "two-ranks", "g6-shape"],
    )
    def test_equal_counts_and_near_equal_tokens(self, lengths, dp):
        order = balanced_dp_order(lengths, dp)
        assert sorted(order) == list(range(len(lengths)))
        balanced = _slice_sums(lengths, order, dp)
        contiguous = _slice_sums(lengths, list(range(len(lengths))), dp)
        assert max(balanced) <= max(contiguous)
        assert max(balanced) - min(balanced) <= max(lengths)

    def test_uneven_split_is_rejected(self):
        with pytest.raises(ValueError):
            balanced_dp_order([1, 2, 3], 2)


def _trainer(dp, *, from_update_forward=True):
    t = RayPPOTrainer.__new__(RayPPOTrainer)
    t.cfg = SimpleNamespace(
        trainer=SimpleNamespace(
            recompute_old_logprobs_per_minibatch=True, old_logprobs_from_update_forward=from_update_forward
        )
    )
    t.dispatch = SimpleNamespace(dp_size=lambda model: dp)
    return t


def _batch(lengths):
    n, width = len(lengths), max(lengths)
    mask = torch.zeros(n, width, dtype=torch.long)
    for i, length in enumerate(lengths):
        mask[i, :length] = 1
    rows = torch.arange(n)
    batch = TrainingInputBatch(
        {
            "sequences": rows[:, None].expand(n, width).clone(),
            "attention_mask": mask,
            "advantages": rows[:, None].float().expand(n, 3).clone(),
        }
    )
    batch.metadata = {"response_length": 3}
    return batch


class TestBalanceDpTokens:
    def test_rows_move_whole_within_their_mini_batch(self):
        lengths = [40, 39, 38, 37, 5, 4, 3, 2] + [30, 1, 29, 2, 28, 3, 27, 4]
        boundaries = [(0, 8), (8, 16)]
        out = _trainer(2)._balance_dp_tokens("policy", _batch(lengths), boundaries)
        order = out["sequences"][:, 0].tolist()
        assert sorted(order[:8]) == list(range(8)) and sorted(order[8:]) == list(range(8, 16))
        for pos, row in enumerate(order):
            assert torch.equal(out["advantages"][pos], torch.full((3,), float(row)))
            assert int(out["attention_mask"][pos].sum()) == lengths[row]
        sums = [sum(lengths[r] for r in order[i : i + 4]) for i in range(0, 16, 4)]
        assert sums[0] == sums[1] == 84
        assert out.metadata == {"response_length": 3}

    def test_rejects_a_separate_old_logprob_forward(self):
        with pytest.raises(ValueError, match="old_logprobs_from_update_forward"):
            _trainer(2, from_update_forward=False)._balance_dp_tokens("policy", _batch([3, 2, 1, 4]), [(0, 4)])
