"""Save and load per-rank RNG state through native Megatron dist checkpointing on CPU.

Each case spawns gloo ranks that form one tensor-parallel group, so every rank owns a
different RNG shard while the data-parallel group (used by the fully parallel strategies)
has a single rank.
"""

import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

pytest.importorskip("megatron.core")

import megatron.core.parallel_state as mpu
from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor
from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

from skyrl.backends.skyrl_train.distributed.megatron import (
    megatron_strategy as strategy,
)
from skyrl.backends.skyrl_train.distributed.utils import get_free_port

pytestmark = pytest.mark.megatron


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))

    def sharded_state_dict(self):
        # Replicated across TP ranks; the TP rank is the replica id.
        return {
            "weight": ShardedTensor.from_rank_offsets(
                "weight", self.weight.data, replica_id=mpu.get_tensor_model_parallel_rank()
            )
        }


def _strategy():
    owner = strategy.MegatronStrategy.__new__(strategy.MegatronStrategy)
    owner.megatron_config = SimpleNamespace(async_dist_ckpt_save=False, async_save_prestage_to_cpu=False)
    owner.is_lora = False
    owner.hf_config = None
    owner.save_hf_configs = lambda *args: None
    return owner


def _seed(value: int):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    get_cuda_rng_tracker().set_states({"model-parallel-rng": torch.full((8,), value % 256, dtype=torch.uint8)})


def _rng_snapshot():
    tracker = get_cuda_rng_tracker().get_states()
    return {
        "random": random.getstate(),
        "numpy": np.random.get_state()[1].copy(),
        "torch": torch.get_rng_state(),
        "tracker": {name: state.clone() for name, state in tracker.items()},
    }


def _assert_rng_equal(actual, expected):
    assert actual["random"] == expected["random"]
    np.testing.assert_array_equal(actual["numpy"], expected["numpy"])
    assert torch.equal(actual["torch"], expected["torch"])
    assert actual["tracker"].keys() == expected["tracker"].keys()
    for name, state in expected["tracker"].items():
        assert torch.equal(actual["tracker"][name], state)


def _worker(rank, world_size, port, case, ckpt_dir):
    os.environ.update(MASTER_ADDR="localhost", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    try:
        mpu.initialize_model_parallel(tensor_model_parallel_size=world_size)
        strategy._async_calls = strategy.AsyncCallsQueue()
        case(rank, ckpt_dir)
    finally:
        mpu.destroy_model_parallel()
        dist.destroy_process_group()


def _run(case, ckpt_dir, world_size=2):
    mp.spawn(_worker, args=(world_size, get_free_port(), case, str(ckpt_dir)), nprocs=world_size)


def _save_then_load_restores_each_rank(rank, ckpt_dir):
    owner, model = _strategy(), SimpleNamespace(actor_module=[_Model()])
    _seed(100 + rank)
    saved = _rng_snapshot()
    owner.save_checkpoint(model, ckpt_dir, node_local_rank=rank)
    _seed(999)

    owner.load_checkpoint(model, ckpt_dir)

    _assert_rng_equal(_rng_snapshot(), saved)


def _single_rng_state_still_loads(rank, ckpt_dir):
    owner, model = _strategy(), SimpleNamespace(actor_module=[_Model()])
    _seed(100 + rank)
    rank0_state = [_rng_snapshot()]
    dist.broadcast_object_list(rank0_state, src=0)
    if rank == 0:
        os.makedirs(ckpt_dir)
    dist.barrier()
    # The previous format saved rank 0's RNG state, without tracker states, as common state.
    dist_checkpointing.save(
        {"model": model.actor_module[0].sharded_state_dict(), "rng": owner.get_rng_state()}, ckpt_dir
    )
    _seed(999)
    tracker = _rng_snapshot()["tracker"]

    owner.load_checkpoint(model, ckpt_dir)

    _assert_rng_equal(_rng_snapshot(), {**rank0_state[0], "tracker": tracker})


def _changed_layout_keeps_current_rng(rank, ckpt_dir):
    owner, model = _strategy(), SimpleNamespace(actor_module=[_Model()])
    _seed(999 + rank)
    current = _rng_snapshot()

    owner.load_checkpoint(model, ckpt_dir)

    _assert_rng_equal(_rng_snapshot(), current)


def test_each_rank_restores_its_own_rng_state(tmp_path):
    _run(_save_then_load_restores_each_rank, tmp_path / "checkpoint")


def test_checkpoint_with_single_rng_state_still_loads(tmp_path):
    _run(_single_rng_state_still_loads, tmp_path / "checkpoint")


def test_changed_parallel_layout_keeps_current_rng_state(tmp_path):
    _run(_save_then_load_restores_each_rank, tmp_path / "checkpoint", world_size=1)

    _run(_changed_layout_keeps_current_rng, tmp_path / "checkpoint")
