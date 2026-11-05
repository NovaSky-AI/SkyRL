"""Wait on the async broadcasts of megatron-core's fully-parallel checkpoint load.

``exchange_loaded_tensors_broadcast`` sends shards a rank only receives to discard with
``async_op=True`` and drops the ``Work``. ProcessGroupNCCL keeps the tensors of un-waited works
alive until a later collective on the group, so their receive buffers survive the load and can
OOM the first training step after a resume. This is upstream's function, but it waits on those
works (at most ``_MAX_IN_FLIGHT`` outstanding).
"""

import collections
import inspect

import torch
from loguru import logger

_APPLIED = False
_MAX_IN_FLIGHT = 16


def _exchange_loaded_tensors_broadcast(loaded_tensors, unloaded_shards, shard_distribution, parallelization_group=None):
    from megatron.core.dist_checkpointing import exchange_utils
    from megatron.core.fp8_utils import is_float8tensor

    main_rank_for_shard, _, shard_to_metadata, all_ranks_for_shard = shard_distribution
    local_rank = torch.distributed.get_rank(group=parallelization_group)

    all_loaded_tensors = dict(loaded_tensors)
    pending = collections.deque()

    for shard_id, rank in main_rank_for_shard.items():
        if len(all_ranks_for_shard[shard_id]) == 1:
            assert all_ranks_for_shard[shard_id][0] == main_rank_for_shard[shard_id], (
                f"When there is only 1 ranks that needs a given shard, it should be the loading rank. "
                f"Got: needs [{all_ranks_for_shard[shard_id][0]}] vs loads [{main_rank_for_shard[shard_id]}]"
            )
            continue
        if rank == local_rank:
            assert shard_id in all_loaded_tensors, (shard_id, all_loaded_tensors.keys())
            orig_device = all_loaded_tensors[shard_id].device
            local_ten = all_loaded_tensors[shard_id].cuda()
        else:
            local_ten, orig_device = exchange_utils._get_empty_tensor_for_exchange(
                shard_id, unloaded_shards, shard_to_metadata, all_loaded_tensors
            )

        # Upstream: exchange a nominal dtype instead of FP8 (TE bug); TE handles it at state dict load.
        if is_float8tensor(local_ten):
            try:
                local_ten = local_ten.from_float8()
            except Exception:
                local_ten = local_ten.dequantize()
            all_loaded_tensors[shard_id] = local_ten

        global_src_rank = (
            rank if parallelization_group is None else torch.distributed.get_global_rank(parallelization_group, rank)
        )
        # Upstream: async only when there is no CPU-copy follow-up.
        work = torch.distributed.broadcast(
            local_ten, src=global_src_rank, group=parallelization_group, async_op=orig_device is None
        )
        if work is not None:
            pending.append(work)
            if len(pending) > _MAX_IN_FLIGHT:
                pending.popleft().wait()
        if orig_device is not None:
            all_loaded_tensors[shard_id] = local_ten.to(orig_device)
        del local_ten

    while pending:
        pending.popleft().wait()
    return all_loaded_tensors


def apply_dist_ckpt_exchange_wait_patch() -> None:
    global _APPLIED
    if _APPLIED:
        return
    from megatron.core.dist_checkpointing import exchange_utils

    original = exchange_utils.exchange_loaded_tensors_broadcast
    if original is _exchange_loaded_tensors_broadcast:  # already installed
        _APPLIED = True
        return
    try:
        already_waits = ".wait()" in inspect.getsource(original)
    except (OSError, TypeError):  # source unavailable
        already_waits = False
    if already_waits:
        logger.warning("exchange_loaded_tensors_broadcast already waits; patch_dist_ckpt_exchange_wait.py can go")
        _APPLIED = True
        return
    # exchange_by_distribution looks the function up as a module global at call time.
    _exchange_loaded_tensors_broadcast.__wrapped__ = original
    exchange_utils.exchange_loaded_tensors_broadcast = _exchange_loaded_tensors_broadcast
    _APPLIED = True
    logger.info("dist_checkpointing: fully-parallel load waits on its async exchange broadcasts")
