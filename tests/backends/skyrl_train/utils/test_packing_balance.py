import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.packing_balance import (
    compute_effective_sequence_costs,
    compute_slot_costs,
    invert_permutation,
    plan_balanced_slot_permutation,
)
from tests.train.util import example_dummy_config


def _make_batch_from_lengths(lengths: list[int]) -> TrainingInputBatch:
    max_len = max(lengths)
    attention_mask = torch.zeros(len(lengths), max_len, dtype=torch.int64)
    for row, length in enumerate(lengths):
        attention_mask[row, :length] = 1
    return TrainingInputBatch(
        {
            "sequences": torch.arange(len(lengths) * max_len, dtype=torch.int64).reshape(len(lengths), max_len),
            "attention_mask": attention_mask,
        }
    )


def test_plan_balanced_slot_permutation_is_deterministic_and_invertible():
    costs = [9, 8, 7, 6, 5, 4, 3, 2]
    permutation = plan_balanced_slot_permutation(costs, dp_size=2, local_micro_bsz=2)

    assert permutation == plan_balanced_slot_permutation(costs, dp_size=2, local_micro_bsz=2)
    assert sorted(permutation) == list(range(len(costs)))

    inverse = invert_permutation(permutation)
    restored = [permutation[inverse_idx] for inverse_idx in inverse]
    assert restored == list(range(len(costs)))


def test_plan_balanced_slot_permutation_reduces_max_slot_cost_on_skewed_case():
    costs = [100, 99, 98, 97, 4, 3, 2, 1]

    baseline_slot_costs = compute_slot_costs(costs, dp_size=2, local_micro_bsz=2)
    permutation = plan_balanced_slot_permutation(costs, dp_size=2, local_micro_bsz=2)
    balanced_slot_costs = compute_slot_costs(costs, dp_size=2, local_micro_bsz=2, permutation=permutation)

    assert max(balanced_slot_costs) < max(baseline_slot_costs)


def test_compute_effective_sequence_costs_megatron_uses_alignment():
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 4
    cfg.trainer.policy.megatron_config.context_parallel_size = 2

    batch = _make_batch_from_lengths([1, 16, 17, 31])
    costs = compute_effective_sequence_costs(batch, model="policy", cfg=cfg)

    assert costs == [16, 16, 32, 32]
