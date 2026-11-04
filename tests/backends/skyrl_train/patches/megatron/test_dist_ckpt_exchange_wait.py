"""CPU test for ``patch_dist_ckpt_exchange_wait``: every async exchange broadcast is waited on."""

import pytest
import torch

pytest.importorskip("megatron.core", reason="requires the megatron extra")

# Runs in the CPU megatron job (`-m megatron`); without the marker that job deselects it.
pytestmark = pytest.mark.megatron


class _Work:
    def __init__(self, log):
        self.waited = False
        self._log = log

    def wait(self):
        self.waited = True
        self._log.append(self)


@pytest.fixture
def patched(monkeypatch):
    from megatron.core.dist_checkpointing import exchange_utils

    from skyrl.backends.skyrl_train.patches.megatron import (
        patch_dist_ckpt_exchange_wait as patch,
    )

    monkeypatch.setattr(
        exchange_utils, "exchange_loaded_tensors_broadcast", exchange_utils.exchange_loaded_tensors_broadcast
    )
    monkeypatch.setattr(patch, "_APPLIED", False)
    patch.apply_dist_ckpt_exchange_wait_patch()
    return exchange_utils, patch


def _run(exchange_utils, monkeypatch, num_shards, local_rank=0):
    works, waited, in_flight = [], [], []

    def broadcast(tensor, src, group=None, async_op=False):
        if not async_op:
            return None
        work = _Work(waited)
        works.append(work)
        in_flight.append(sum(not w.waited for w in works))
        return work

    def empty_for_exchange(shard_id, unloaded_shards, shard_to_metadata, all_loaded_tensors):
        return torch.zeros(4), None  # a shard this rank only receives to discard

    monkeypatch.setattr(torch.distributed, "broadcast", broadcast)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: local_rank)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda group, rank: rank)
    monkeypatch.setattr(exchange_utils, "_get_empty_tensor_for_exchange", empty_for_exchange)
    # Every shard is loaded by rank 1 and needed by ranks 1 and 2: rank 0 receives and discards it.
    main_rank = {("s", i): 1 for i in range(num_shards)}
    all_ranks = {("s", i): [1, 2] for i in range(num_shards)}
    out = exchange_utils.exchange_by_distribution(
        {}, {}, (main_rank, None, {}, all_ranks), parallelization_group=None, exchange_algo="broadcast"
    )
    return out, works, in_flight


def test_every_discarded_broadcast_is_waited_with_a_bounded_backlog(patched, monkeypatch):
    exchange_utils, patch = patched
    out, works, in_flight = _run(exchange_utils, monkeypatch, num_shards=100)
    assert out == {}
    assert len(works) == 100
    assert all(w.waited for w in works)
    assert max(in_flight) <= patch._MAX_IN_FLIGHT + 1


def test_apply_is_idempotent_and_wraps_upstream(patched):
    exchange_utils, patch = patched
    first = exchange_utils.exchange_loaded_tensors_broadcast
    patch.apply_dist_ckpt_exchange_wait_patch()
    assert exchange_utils.exchange_loaded_tensors_broadcast is first
    assert first.__wrapped__.__module__ == exchange_utils.__name__
