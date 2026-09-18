"""CPU-only coverage of device selection across broadcast worker threads."""

import asyncio
import threading
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.weight_sync import broadcast_strategy
from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.broadcast_strategy import (
    BroadcastInitInfo,
    BroadcastTransferStrategy,
    BroadcastWeightTransferSender,
)


@pytest.fixture(autouse=True)
def no_ray_cluster(monkeypatch):
    monkeypatch.setattr(ray, "init", Mock(side_effect=AssertionError("Unit tests must not start Ray")))


@pytest.fixture
def cuda_devices(monkeypatch):
    # CUDA's current device belongs to the host thread; a new thread starts at 0.
    state = threading.local()

    def current_device():
        return getattr(state, "device", 0)

    def set_device(device):
        state.device = device.index if isinstance(device, torch.device) else int(device)

    @contextmanager
    def device_context(device):
        previous = current_device()
        set_device(device)
        try:
            yield
        finally:
            set_device(previous)

    monkeypatch.setattr(torch.cuda, "current_device", current_device)
    monkeypatch.setattr(torch.cuda, "set_device", set_device)
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid=f"GPU-{device}"))
    return state


def _init_info(world_size=3):
    return BroadcastInitInfo(
        master_addr="127.0.0.1",
        master_port=12345,
        rank_offset=1,
        world_size=world_size,
        override_existing_receiver=False,
    )


@pytest.mark.parametrize("device", [0, 2, 5])
def test_sender_thread_uses_trainer_device(monkeypatch, cuda_devices, device):
    monkeypatch.setenv("LOCAL_RANK", str(device))
    torch.cuda.set_device(device)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(
        broadcast_strategy,
        "nccl_trainer_init",
        lambda info: SimpleNamespace(device=torch.cuda.current_device()),
    )

    sender = asyncio.run(asyncio.to_thread(BroadcastTransferStrategy.create_sender, _init_info(), AsyncMock()))

    assert sender._model_update_group.device == device
    assert torch.cuda.current_device() == device


@pytest.mark.parametrize("derive_metadata", [False, True])
def test_each_send_thread_uses_communicator_device(monkeypatch, cuda_devices, derive_metadata):
    torch.cuda.set_device(2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    group = SimpleNamespace(device=torch.device("cuda:2"))
    client = AsyncMock()
    sender = BroadcastWeightTransferSender(_init_info(), group, client)
    sent = []

    def send_weights(weights, communicator, *, packed):
        sent.append((torch.cuda.current_device(), list(weights), communicator, packed))

    monkeypatch.setattr(broadcast_strategy, "nccl_trainer_send_weights", send_weights)
    tensor = torch.zeros(1)
    chunk = WeightChunk(names=["w"], dtypes=["float32"], shapes=[[1]], tensors=[tensor])
    metadata = None if derive_metadata else {"names": ["w"], "dtype_names": ["float32"], "shapes": [[1]]}

    # Separate event loops get fresh executor threads; initialization cannot have selected their device.
    for _ in range(2):
        asyncio.run(sender.send_chunks([chunk], weight_metadata=metadata, derive_metadata_from_chunks=derive_metadata))

    assert sent == [(2, [("w", tensor)], group, True)] * 2
    assert torch.cuda.current_device() == 2


def _mock_collectives(monkeypatch, trainer_uuids, rank=0):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: len(trainer_uuids))

    def all_gather_object(output, uuid):
        assert uuid == trainer_uuids[rank]
        output[:] = trainer_uuids

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    monkeypatch.setattr(torch.distributed, "broadcast_object_list", lambda error, src: None)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(ray._private.services, "get_node_ip_address", lambda: "127.0.0.1")


def _worker(colocate_all=False):
    from skyrl.backends.skyrl_train.workers.worker import Worker

    worker = Worker.__new__(Worker)
    worker.cfg = SimpleNamespace(
        placement=SimpleNamespace(colocate_all=colocate_all),
        policy=SimpleNamespace(model=SimpleNamespace(path="unused")),
    )
    return worker


def _client(inference_uuids, world_size=None):
    client = AsyncMock()
    world_size = sum(map(len, inference_uuids.values())) if world_size is None else world_size
    client.get_world_size.return_value = (world_size, world_size)
    client.get_gpu_uuids.return_value = inference_uuids
    return client


_ENGINE_CFG = SimpleNamespace(weight_sync_backend="nccl", run_engines_locally=True)


@pytest.mark.parametrize("engines,dp,tp", [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 2, 2)])
@pytest.mark.parametrize("noset", [False, True])
def test_ray_assigned_trainer_device_survives_weight_sync_init(monkeypatch, cuda_devices, engines, dp, tp, noset):
    from skyrl.backends.skyrl_train.workers import worker as worker_module
    from skyrl.train.utils import ray_logging

    trainer_device = engines * dp * tp
    local_device = trainer_device if noset else 0
    if noset:
        monkeypatch.setenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "1")
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(trainer_device + 1))))
    else:
        monkeypatch.delenv("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(trainer_device))

        def get_device_properties(device):
            assert device == 0
            return SimpleNamespace(uuid=f"GPU-{trainer_device}")

        monkeypatch.setattr(torch.cuda, "get_device_properties", get_device_properties)
    monkeypatch.setattr(ray, "get_gpu_ids", lambda: [trainer_device])
    monkeypatch.setattr(worker_module, "configure_ray_worker_logging", lambda: None)
    monkeypatch.setattr(ray_logging, "redirect_actor_output_to_file", lambda: None)
    worker = _worker()
    worker_module.DistributedTorchRayActor.__init__(
        worker,
        world_size=1,
        rank=0,
        local_rank=0,
        master_addr="127.0.0.1",
        master_port=12345,
        sequence_parallel_size=1,
    )
    _mock_collectives(monkeypatch, [f"GPU-{trainer_device}"])
    client = _client({f"server-{i}": [f"GPU-{i * tp + j}" for j in range(tp)] for i in range(engines * dp)})

    def init_sender(info):
        client.get_gpu_uuids.assert_awaited_once()
        assert info["world_size"] == trainer_device + 1
        return SimpleNamespace(device=torch.cuda.current_device())

    monkeypatch.setattr(broadcast_strategy, "nccl_trainer_init", init_sender)
    asyncio.run(worker.init_weight_sync_state(client, _ENGINE_CFG))

    assert worker._weight_transfer_sender._model_update_group.device == local_device
    client.init_weight_update_communicator.assert_awaited_once()


@pytest.mark.parametrize(
    "trainer_uuids,inference_uuids,participants",
    [
        (["GPU-2"], {"server-0": ["GPU-2", "GPU-1"]}, ["trainer rank 0", "inference worker 0 on server-0"]),
        (["GPU-2", "GPU-3"], {"server-0": ["GPU-3"]}, ["trainer rank 1", "inference worker 0 on server-0"]),
        (["GPU-2"], {"server-0": ["GPU-0", "GPU-0"]}, ["inference worker 0", "inference worker 1"]),
        (["GPU-2"], {"server-0": ["GPU-0"], "server-1": ["GPU-0"]}, ["server-0", "server-1"]),
        (["GPU-2", "GPU-2"], {"server-0": ["GPU-0"]}, ["trainer rank 0", "trainer rank 1"]),
    ],
)
def test_duplicate_uuid_prevents_both_communicator_initializers(
    monkeypatch, cuda_devices, trainer_uuids, inference_uuids, participants
):
    monkeypatch.setenv("LOCAL_RANK", "2")
    _mock_collectives(monkeypatch, trainer_uuids)
    init_sender = Mock()
    monkeypatch.setattr(broadcast_strategy, "nccl_trainer_init", init_sender)
    client = _client(inference_uuids)

    with pytest.raises(RuntimeError, match="Duplicate physical GPU UUID") as exc:
        asyncio.run(_worker().init_weight_sync_state(client, _ENGINE_CFG))

    assert all(participant in str(exc.value) for participant in participants)
    init_sender.assert_not_called()
    client.init_weight_update_communicator.assert_not_awaited()


@pytest.mark.parametrize("failure", ["count", "rpc"])
def test_incomplete_uuid_report_prevents_communicator_initialization(monkeypatch, cuda_devices, failure):
    monkeypatch.setenv("LOCAL_RANK", "2")
    _mock_collectives(monkeypatch, ["GPU-2"])
    init_sender = Mock()
    monkeypatch.setattr(broadcast_strategy, "nccl_trainer_init", init_sender)
    client = _client({"server-0": ["GPU-0"]}, world_size=2)
    if failure == "rpc":
        client.get_gpu_uuids.side_effect = RuntimeError("UUID RPC failed")

    with pytest.raises(RuntimeError, match="Expected 2 inference GPU UUIDs|UUID RPC failed"):
        asyncio.run(_worker().init_weight_sync_state(client, _ENGINE_CFG))

    init_sender.assert_not_called()
    client.init_weight_update_communicator.assert_not_awaited()


def test_validation_failure_reaches_nonzero_trainer_ranks(monkeypatch, cuda_devices):
    client = _client({"server-0": ["GPU-3"]})
    validation_error = []

    def broadcast_object_list(error, src):
        assert src == 0
        if not validation_error:
            validation_error[:] = error
        else:
            error[:] = validation_error

    for rank in range(2):
        monkeypatch.setenv("LOCAL_RANK", str(rank + 2))
        _mock_collectives(monkeypatch, ["GPU-2", "GPU-3"], rank=rank)
        monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast_object_list)
        with pytest.raises(RuntimeError, match="Duplicate physical GPU UUID 'GPU-3'"):
            asyncio.run(_worker().init_weight_sync_state(client, _ENGINE_CFG))

    client.get_gpu_uuids.assert_awaited_once()
    client.init_weight_update_communicator.assert_not_awaited()


def test_distinct_physical_gpus_with_same_node_local_ordinal_are_allowed(monkeypatch, cuda_devices):
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(uuid="GPU-node-a-0"))
    _mock_collectives(monkeypatch, ["GPU-node-a-0", "GPU-node-b-0"])
    client = _client({"server-node-c": ["GPU-node-c-0"], "server-node-d": ["GPU-node-d-0"]})

    asyncio.run(BroadcastTransferStrategy.validate_placement(client, inference_world_size=2))

    client.get_gpu_uuids.assert_awaited_once()


def test_colocated_init_keeps_cuda_ipc_path(monkeypatch, cuda_devices):
    from skyrl.backends.skyrl_train.weight_sync import CudaIpcTransferStrategy

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    sender = object()
    monkeypatch.setattr(CudaIpcTransferStrategy, "create_init_info", lambda *args, **kwargs: object())
    monkeypatch.setattr(CudaIpcTransferStrategy, "create_sender", lambda **kwargs: sender)
    validate = AsyncMock()
    monkeypatch.setattr(BroadcastTransferStrategy, "validate_placement", validate)
    client = _client({"server-0": ["GPU-0"]})
    worker = _worker(colocate_all=True)

    asyncio.run(worker.init_weight_sync_state(client, _ENGINE_CFG))

    assert worker._weight_transfer_sender is sender
    validate.assert_not_awaited()
    client.get_gpu_uuids.assert_not_awaited()
    client.init_weight_update_communicator.assert_awaited_once()
