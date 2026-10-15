"""Keep Core's checkpoint index map aligned with its mixed-dtype optimizer groups.

DistributedOptimizer builds its model-to-optimizer index map in gradient-buffer
order, then constructs each optimizer group with FP32 shards before FP16/BF16
shards. Mixed groups can therefore point at a different parameter's optimizer
state. Unequal shard sizes fail checkpointing; equal sizes can silently exchange
states. Loading also uses this map, including the constructor's initial state
allocation, so correcting it at the save boundary is too late.

Order the local model-parameter groups before Core creates the shards. This does
not reorder the DDP buffers, cast parameters, or change optimizer hyperparameters.
Remove this adapter once the pinned Core builds its index map in shard order.
"""

from functools import wraps

import torch


def make_fp32_first_group_ranges(original):
    """Match Core's stable FP32-then-low-precision shard concatenation."""

    @wraps(original)
    def build_group_ranges(cls, param_groups, gbuf_ranges):
        index_map, group_ranges = original(cls, param_groups, gbuf_ranges)
        for group_index, group in enumerate(group_ranges):
            params = group["params"]
            # Python's sort is stable: preserve order within each dtype family.
            ordered = sorted(params, key=lambda param: param.dtype != torch.float32)
            group["params"] = ordered
            for param_index, param in enumerate(ordered):
                index_map[param] = (group_index, param_index)
        return index_map, group_ranges

    build_group_ranges._skyrl_optimizer_group_order = True
    return build_group_ranges


def patch_optimizer_group_order() -> None:
    """Install before DistributedOptimizer allocates or restores any state."""
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer

    original = DistributedOptimizer._build_optimizer_group_ranges.__func__
    if getattr(original, "_skyrl_optimizer_group_order", False):
        return
    DistributedOptimizer._build_optimizer_group_ranges = classmethod(make_fp32_first_group_ranges(original))
