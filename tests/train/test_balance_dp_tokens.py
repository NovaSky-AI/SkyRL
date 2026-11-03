"""
Tests for ``trainer.balance_dp_tokens``.

uv run --isolated --extra dev pytest tests/train/test_balance_dp_tokens.py
"""

from unittest.mock import MagicMock

import pytest
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.packed_tensor import PackedTensor
from skyrl.train.dataset.preprocess import balance_dp_token_permutation
from skyrl.train.trainer import RayPPOTrainer
from tests.train.util import example_dummy_config


def _chunk_loads(token_counts, perm, boundaries, dp_size):
    loads = []
    for start, end in boundaries:
        per_rank = (end - start) // dp_size
        rows = perm[start:end]
        loads.append([sum(token_counts[r] for r in rows[i * per_rank : (i + 1) * per_rank]) for i in range(dp_size)])
    return loads


def test_permutation_balances_within_each_minibatch():
    # Two mini-batches of 8 rows; in prompt order the long rows sit on the first DP chunk.
    token_counts = [8000, 8000, 7000, 7000, 500, 500, 600, 600] + [9000, 9000, 100, 100, 200, 200, 300, 300]
    boundaries = [(0, 8), (8, 16)]
    perm = balance_dp_token_permutation(token_counts, boundaries, dp_size=4)

    assert sorted(perm[0:8]) == list(range(0, 8)) and sorted(perm[8:16]) == list(range(8, 16))
    before = _chunk_loads(token_counts, list(range(16)), boundaries, 4)
    after = _chunk_loads(token_counts, perm, boundaries, 4)
    for mb_before, mb_after in zip(before, after):
        assert max(mb_after) < max(mb_before)
        assert sum(mb_after) == sum(mb_before)
    assert max(after[0]) == 8500  # 8000+500 / 8000+500 / 7000+600 / 7000+600


@pytest.mark.parametrize(
    "token_counts, boundaries, dp_size",
    [
        ([5, 1, 1, 1], [(0, 4)], 1),  # single DP rank
        ([5, 1, 1, 1, 9, 9], [(0, 4), (4, 6)], 4),  # a mini-batch staging would pad
        ([3, 3, 3, 3], [(0, 4)], 2),  # already balanced
    ],
)
def test_permutation_none_when_nothing_to_do(token_counts, boundaries, dp_size):
    assert balance_dp_token_permutation(token_counts, boundaries, dp_size) is None


def test_index_select_reorders_every_field_type():
    data = TrainingInputBatch(
        {
            "sequences": torch.arange(6).view(3, 2),
            "routes": PackedTensor.from_segments([torch.full((n, 2), n) for n in (1, 2, 3)]),
        }
    )
    data.metadata = {"k": 1}
    out = data.index_select([2, 0, 1])
    assert torch.equal(out["sequences"], torch.tensor([[4, 5], [0, 1], [2, 3]]))
    assert [len(out["routes"].segment(i)) for i in range(3)] == [3, 1, 2]
    assert out.metadata == {"k": 1}


class _DummyDataset:
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return "dummy"

    def collate_fn(self, batch):
        return batch


def test_forward_pass_restores_original_row_order():
    cfg = example_dummy_config()
    cfg.trainer.balance_dp_tokens = True
    cfg.trainer.recompute_old_logprobs_per_minibatch = True
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=None,
        tokenizer=tokenizer,
        train_dataset=_DummyDataset(),
        eval_dataset=None,
        inference_engine_client=None,
        generator=MagicMock(),
    )
    trainer.dispatch = MagicMock()
    trainer.dispatch.dp_size.return_value = 2

    lengths = [9, 1, 8, 2]
    attention_mask = torch.zeros(4, 10, dtype=torch.long)
    for i, n in enumerate(lengths):
        attention_mask[i, :n] = 1
    data = TrainingInputBatch(
        {"sequences": torch.arange(4).unsqueeze(1).repeat(1, 10), "attention_mask": attention_mask}
    )
    staged_rows = []

    def stage_data(model, staged, boundaries):
        staged_rows.append(staged["sequences"][:, 0].tolist())
        return [[staged]]

    def forward_from_staged(model, chunk_refs):
        rows = chunk_refs[0]["sequences"][:, 0].tolist()
        return WorkerOutput(loss_fn_outputs=[{"logprobs": [float(r)] * 2} for r in rows])

    trainer.dispatch.stage_data.side_effect = stage_data
    trainer.dispatch.forward_from_staged.side_effect = forward_from_staged

    out = trainer._execute_forward_pass("policy", data, key="logprobs", mini_batch_boundaries=[(0, 4)])

    # Contiguous chunks carry 10 and 10 tokens only by luck of order; [9, 1, 8, 2] is staged as {9, 1}, {8, 2}.
    first, second = staged_rows[0][:2], staged_rows[0][2:]
    assert sum(lengths[r] for r in first) == sum(lengths[r] for r in second) == 10
    assert torch.equal(out[:, 0], torch.tensor([0.0, 1.0, 2.0, 3.0]))
