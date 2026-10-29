"""CPU tests for ``async_save_streaming_write`` with real ``torch_dist`` saves and loads.

The checkpoints are written and loaded through Megatron's ``dist_checkpointing`` on a
gloo process group, with CPU tensors in place of GPU tensors.

Run with:
RAY_ADDRESS=local uv run --isolated --extra dev --extra megatron -- pytest -s \
    tests/backends/skyrl_train/distributed/test_megatron_streaming_checkpoint.py
"""

import contextlib
import multiprocessing
import os
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist

pytest.importorskip("megatron.core")

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing import ShardedTensor
from torch.distributed.checkpoint import CheckpointException
from torch.distributed.checkpoint.filesystem import FileSystem

from skyrl.backends.skyrl_train.distributed.megatron import (
    megatron_strategy,
    streaming_checkpoint,
)
from skyrl.backends.skyrl_train.distributed.utils import get_free_port
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)
from skyrl.train.config.config import MegatronConfig, MegatronDDPConfig

pytestmark = pytest.mark.megatron

_TIMEOUT_S = 60


@pytest.fixture(scope="module")
def gloo_group():
    """Single-rank gloo group. Destroy it only if this fixture created it."""
    initialized_here = False
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        initialized_here = True
    yield dist.group.WORLD
    if initialized_here and dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def cpu_finalize(monkeypatch):
    """NVRx's save finalization puts its failure flag on the current CUDA device."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")


def _tensors(seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    flat = torch.randn(64, generator=generator)
    return {
        "weight": torch.randn(8, 4, generator=generator).to(torch.bfloat16),
        "bias": torch.randn(4, generator=generator),
        "steps": torch.arange(5, dtype=torch.int64),
        # A view into a larger buffer, like the distributed optimizer state.
        "exp_avg": flat[8:40].view(4, 8),
    }


def _sharded(tensors: dict) -> dict:
    return {name: ShardedTensor.from_rank_offsets(name, tensor, replica_id=0) for name, tensor in tensors.items()}


def _load(checkpoint_dir, like: dict) -> dict:
    template = {"model": _sharded({name: torch.empty_like(tensor) for name, tensor in like.items()})}
    return dist_checkpointing.load(template, str(checkpoint_dir))["model"]


def _assert_bytes_equal(actual: dict, expected: dict) -> None:
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert actual[name].dtype == tensor.dtype, name
        assert torch.equal(actual[name].contiguous().view(torch.uint8), tensor.contiguous().view(torch.uint8)), name


def _streaming_strategy(**config) -> megatron_strategy.MegatronStrategy:
    megatron_config = MegatronConfig(async_dist_ckpt_save=True, async_save_streaming_write=True, **config)
    strategy = megatron_strategy.MegatronStrategy(megatron_config)
    strategy.is_rank_0 = Mock(return_value=False)
    strategy.print = Mock()
    return strategy


def _model(tensors: dict):
    module = SimpleNamespace(sharded_state_dict=lambda: _sharded(tensors))
    return SimpleNamespace(actor_module=[module])


def _mutate(tensors: dict) -> None:
    for tensor in tensors.values():
        tensor.add_(1)


class _Optimizer:
    def __init__(self, tensors: dict):
        self.tensors = tensors

    def step(self):
        _mutate(self.tensors)
        return True, None, None

    def zero_grad(self):
        pass


@pytest.fixture
def save_strategy(gloo_group, monkeypatch):
    """Patch the parallel-state and device calls that `save_checkpoint` and offload make."""
    monkeypatch.setattr(megatron_strategy.mpu, "get_data_parallel_group", lambda **_: gloo_group)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return _streaming_strategy()


@pytest.fixture
def write_gate(monkeypatch):
    """Hold every streaming write before its first byte until ``release`` is set."""
    gate = SimpleNamespace(started=threading.Event(), release=threading.Event())
    write_data = streaming_checkpoint._LiveTensorFileSystemWriter.write_data

    def gated_write_data(self, plan, planner):
        gate.started.set()
        assert gate.release.wait(timeout=_TIMEOUT_S)
        return write_data(self, plan, planner)

    monkeypatch.setattr(streaming_checkpoint._LiveTensorFileSystemWriter, "write_data", gated_write_data)
    yield gate
    gate.release.set()


def test_streaming_write_loads_back_byte_equal(gloo_group, tmp_path):
    tensors = _tensors(seed=0)
    request = dist_checkpointing.save(
        {"model": _sharded(tensors), "lr_scheduler": {"num_steps": 3}},
        str(tmp_path),
        sharded_strategy=streaming_checkpoint.StreamingTorchDistSaveShardedStrategy(),
        async_sharded_save=True,
    )
    assert request.preload_fn is None

    streaming_checkpoint.StreamingCheckpointWrite(request).finish()

    _assert_bytes_equal(_load(tmp_path, tensors), tensors)
    common = dist_checkpointing.load_common_state_dict(str(tmp_path))
    assert common["lr_scheduler"] == {"num_steps": 3}


def _offload(strategy, tensors):
    strategy.offload_to_cpu(model=None, optimizer=None)


def _backload(strategy, tensors):
    strategy.backload_to_gpu(model=None, optimizer=None)


def _optimizer_step(strategy, tensors):
    strategy.optimizer_step(_Optimizer(tensors), model=None, scheduler=Mock())


def _finalize_then_mutate(strategy, tensors):
    strategy.finalize_pending_saves()
    _mutate(tensors)


@pytest.mark.parametrize(
    "mutating_call",
    [_optimizer_step, _offload, _backload, _finalize_then_mutate],
    ids=["optimizer_step", "offload_to_cpu", "backload_to_gpu", "finalize_pending_saves"],
)
def test_mutating_calls_wait_for_streaming_write(save_strategy, write_gate, tmp_path, monkeypatch, mutating_call):
    tensors = _tensors(seed=1)
    expected = {name: tensor.clone() for name, tensor in tensors.items()}
    # Offload and backload change the live tensors, like a release of their storage.
    for name in ("offload_megatron_model_to_cpu", "load_megatron_model_to_gpu"):
        monkeypatch.setattr(megatron_strategy, name, lambda *_, **__: _mutate(tensors))
    for name in (
        "offload_megatron_grads_to_cpu",
        "offload_megatron_optimizer",
        "load_megatron_grads_to_gpu",
        "load_megatron_optimizer",
    ):
        monkeypatch.setattr(megatron_strategy, name, lambda *_, **__: None)

    save_strategy.save_checkpoint(_model(tensors), str(tmp_path), node_local_rank=0)
    assert write_gate.started.wait(timeout=_TIMEOUT_S)
    caller = threading.Thread(target=mutating_call, args=(save_strategy, tensors))
    caller.start()
    caller.join(timeout=1.0)

    assert caller.is_alive(), "the call returned before the checkpoint write ended"
    _assert_bytes_equal(tensors, expected)
    assert not (tmp_path / "metadata.json").exists()

    write_gate.release.set()
    caller.join(timeout=_TIMEOUT_S)
    assert not caller.is_alive()
    _assert_bytes_equal(_load(tmp_path, tensors), expected)
    assert not torch.equal(tensors["bias"], expected["bias"])


def test_optim_step_waits_before_grad_sync(save_strategy, write_gate, tmp_path):
    """The grad sync of ``optim_step`` updates the MoE router expert bias in place."""
    tensors = {"router.expert_bias": torch.zeros(4)}
    expected = {name: tensor.clone() for name, tensor in tensors.items()}
    worker = MegatronPolicyWorkerBase.__new__(MegatronPolicyWorkerBase)
    worker.strategy = save_strategy
    worker.optimizer = _Optimizer({})
    worker.scheduler = Mock()
    worker.actor_module = []
    worker.model = SimpleNamespace(run_pending_grad_sync=lambda: _mutate(tensors))

    save_strategy.save_checkpoint(_model(tensors), str(tmp_path), node_local_rank=0)
    assert write_gate.started.wait(timeout=_TIMEOUT_S)
    caller = threading.Thread(target=worker.optim_step)
    caller.start()
    caller.join(timeout=1.0)
    assert caller.is_alive(), "the grad sync ran before the checkpoint write ended"

    write_gate.release.set()
    caller.join(timeout=_TIMEOUT_S)
    assert not caller.is_alive()
    _assert_bytes_equal(_load(tmp_path, tensors), expected)
    assert torch.equal(tensors["router.expert_bias"], expected["router.expert_bias"] + 1)


@contextlib.contextmanager
def _failing_create_stream(self, path, mode):
    raise OSError(28, "No space left on device")
    yield


def test_write_error_surfaces_at_fence_without_metadata(save_strategy, tmp_path, monkeypatch):
    tensors = _tensors(seed=2)
    expected = {name: tensor.clone() for name, tensor in tensors.items()}
    monkeypatch.setattr(FileSystem, "create_stream", _failing_create_stream)

    save_strategy.save_checkpoint(_model(tensors), str(tmp_path), node_local_rank=0)
    with pytest.raises(CheckpointException, match="No space left on device"):
        save_strategy.optimizer_step(_Optimizer(tensors), model=None, scheduler=Mock())

    _assert_bytes_equal(tensors, expected)
    assert not (tmp_path / ".metadata").exists()
    assert not (tmp_path / "metadata.json").exists()
    # The failed write is not pending anymore.
    save_strategy.finish_streaming_save()


def _two_rank_worker(rank: int, port: int, checkpoint_root: str, result_dir: str) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=2)
    torch.cuda.current_device = lambda: "cpu"
    result = "ok"
    try:
        local = {name: tensor + rank for name, tensor in _tensors(seed=3).items()}
        sharded = {
            name: ShardedTensor.from_rank_offsets(name, tensor, (0, rank, 2), replica_id=0)
            for name, tensor in local.items()
        }

        def save(checkpoint_dir):
            os.makedirs(checkpoint_dir, exist_ok=True)
            dist.barrier()
            request = dist_checkpointing.save(
                {"model": sharded},
                checkpoint_dir,
                sharded_strategy=streaming_checkpoint.StreamingTorchDistSaveShardedStrategy(),
                async_sharded_save=True,
            )
            streaming_checkpoint.StreamingCheckpointWrite(request).finish()

        good_dir = os.path.join(checkpoint_root, "good")
        save(good_dir)
        template = {
            name: ShardedTensor.from_rank_offsets(name, torch.empty_like(tensor), (0, rank, 2), replica_id=0)
            for name, tensor in local.items()
        }
        _assert_bytes_equal(dist_checkpointing.load({"model": template}, good_dir)["model"], local)

        if rank == 1:
            FileSystem.create_stream = _failing_create_stream
        try:
            save(os.path.join(checkpoint_root, "failed"))
            result = "no error"
        except CheckpointException:
            result = "CheckpointException"
    except BaseException as exc:
        result = f"unexpected {exc!r}"
    finally:
        with open(os.path.join(result_dir, f"rank{rank}"), "w") as f:
            f.write(result)
        dist.destroy_process_group()


def test_write_error_on_one_rank_fails_every_rank(tmp_path):
    checkpoint_root = tmp_path / "ckpt"
    result_dir = tmp_path / "results"
    checkpoint_root.mkdir()
    result_dir.mkdir()
    context = multiprocessing.get_context("spawn")
    port = get_free_port()
    processes = [
        context.Process(target=_two_rank_worker, args=(rank, port, str(checkpoint_root), str(result_dir)))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=4 * _TIMEOUT_S)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.kill()
    assert not alive, "a rank did not finish: the failure did not reach every rank"

    results = {rank: (result_dir / f"rank{rank}").read_text() for rank in range(2)}
    assert results == {0: "CheckpointException", 1: "CheckpointException"}
    failed_dir = checkpoint_root / "failed"
    assert not (failed_dir / ".metadata").exists()
    assert not (failed_dir / "metadata.json").exists()


def test_streaming_write_rejects_unsupported_configs():
    with pytest.raises(ValueError, match="requires async_dist_ckpt_save"):
        megatron_strategy.MegatronStrategy(MegatronConfig(async_save_streaming_write=True))
    with pytest.raises(ValueError, match="mutually exclusive"):
        _streaming_strategy(async_save_prestage_to_cpu=True)
    with pytest.raises(ValueError, match="overlap_param_gather"):
        _streaming_strategy(ddp_config=MegatronDDPConfig(overlap_param_gather=True))
    with pytest.raises(ValueError, match="LoRA"):
        megatron_strategy.MegatronStrategy(
            MegatronConfig(async_dist_ckpt_save=True, async_save_streaming_write=True), is_lora=True
        )
