"""Helpers for balancing packed-sequence workloads before dispatch."""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import MegatronConfig, SkyRLTrainConfig


@dataclass
class _Slot:
    rank: int
    slot: int
    remaining_capacity: int
    total_cost: int = 0
    sample_indices: List[int] = field(default_factory=list)


def _get_megatron_config(cfg: SkyRLTrainConfig, model: str) -> MegatronConfig:
    if model == "ref":
        return cfg.trainer.ref.megatron_config
    return cfg.trainer.policy.megatron_config


def get_megatron_alignment_size(cfg: SkyRLTrainConfig, model: str) -> int:
    megatron_config = _get_megatron_config(cfg, model)
    tp_size = megatron_config.tensor_model_parallel_size
    cp_size = megatron_config.context_parallel_size
    return tp_size * cp_size * 2 if cp_size > 1 else tp_size


def get_real_sequence_lengths(data: TrainingInputBatch) -> List[int]:
    attention_mask = data.get("attention_mask")
    if attention_mask is None:
        raise ValueError("Packed-sequence balancing requires attention_mask to be present.")
    assert isinstance(attention_mask, torch.Tensor), "attention_mask must be a tensor"
    return attention_mask.sum(dim=-1, dtype=torch.int64).tolist()


def compute_effective_sequence_costs(data: TrainingInputBatch, model: str, cfg: SkyRLTrainConfig) -> List[int]:
    lengths = get_real_sequence_lengths(data)
    if cfg.trainer.strategy != "megatron":
        return lengths

    align_size = get_megatron_alignment_size(cfg, model)
    return [math.ceil(length / align_size) * align_size for length in lengths]


def plan_balanced_slot_permutation(costs: List[int], dp_size: int, local_micro_bsz: int) -> List[int]:
    if len(costs) == 0:
        return []
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    if local_micro_bsz <= 0:
        raise ValueError(f"local_micro_bsz must be positive, got {local_micro_bsz}")
    if len(costs) % dp_size != 0:
        raise ValueError(f"Batch size {len(costs)} must be divisible by dp_size {dp_size}.")

    chunk_size = len(costs) // dp_size
    if chunk_size == 0:
        return []

    slot_capacities: List[int] = [local_micro_bsz] * (chunk_size // local_micro_bsz)
    if chunk_size % local_micro_bsz != 0:
        slot_capacities.append(chunk_size % local_micro_bsz)

    slots = [
        _Slot(rank=rank, slot=slot, remaining_capacity=capacity)
        for rank in range(dp_size)
        for slot, capacity in enumerate(slot_capacities)
    ]

    sorted_indices = sorted(range(len(costs)), key=lambda idx: (-costs[idx], idx))
    for sample_idx in sorted_indices:
        candidate_slots = [slot for slot in slots if slot.remaining_capacity > 0]
        chosen_slot = min(candidate_slots, key=lambda slot: (slot.total_cost, slot.rank, slot.slot))
        chosen_slot.sample_indices.append(sample_idx)
        chosen_slot.remaining_capacity -= 1
        chosen_slot.total_cost += costs[sample_idx]

    permutation: List[int] = []
    for rank in range(dp_size):
        rank_slots = [slot for slot in slots if slot.rank == rank]
        for slot in rank_slots:
            permutation.extend(slot.sample_indices)
    return permutation


def invert_permutation(permutation: List[int]) -> List[int]:
    inverse = [0] * len(permutation)
    for new_idx, old_idx in enumerate(permutation):
        inverse[old_idx] = new_idx
    return inverse


def compute_slot_costs(
    costs: List[int],
    dp_size: int,
    local_micro_bsz: int,
    permutation: Optional[List[int]] = None,
) -> List[int]:
    if len(costs) == 0:
        return []
    if len(costs) % dp_size != 0:
        raise ValueError(f"Batch size {len(costs)} must be divisible by dp_size {dp_size}.")

    ordered_costs = costs if permutation is None else [costs[idx] for idx in permutation]
    chunk_size = len(ordered_costs) // dp_size
    if chunk_size == 0:
        return []

    slot_costs: List[int] = []
    for rank in range(dp_size):
        start = rank * chunk_size
        end = start + chunk_size
        rank_costs = ordered_costs[start:end]
        for slot_start in range(0, chunk_size, local_micro_bsz):
            slot_costs.append(sum(rank_costs[slot_start : slot_start + local_micro_bsz]))
    return slot_costs


def compute_slot_cost_stats(
    costs: List[int],
    dp_size: int,
    local_micro_bsz: int,
    permutation: Optional[List[int]] = None,
) -> dict[str, float]:
    slot_costs = compute_slot_costs(costs, dp_size, local_micro_bsz, permutation=permutation)
    if not slot_costs:
        return {
            "max_slot_cost": 0.0,
            "mean_slot_cost": 0.0,
        }
    return {
        "max_slot_cost": float(max(slot_costs)),
        "mean_slot_cost": float(sum(slot_costs) / len(slot_costs)),
    }
