"""The sender and receiver must decode each GPU's own expert names and buffer sizes."""

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from skyrl.backends.skyrl_train.inference_servers.new_inference_worker_wrap import (
    NewInferenceWorkerWrap,
)
from skyrl.backends.skyrl_train.weight_sync import cuda_ipc_strategy
from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.cuda_ipc_strategy import (
    CudaIpcInitInfo,
    CudaIpcWeightTransferSender,
)


def _rebuild_cpu(values, _1, _2, _3, _4, _5, device):
    assert device in (0, 1)
    return torch.tensor(values, dtype=torch.uint8)


@pytest.mark.parametrize("receiver_rank", [0, 1])
def test_ipc_uses_metadata_from_the_handle_owning_gpu(monkeypatch, receiver_rank):
    client = AsyncMock()
    sender = CudaIpcWeightTransferSender(
        CudaIpcInitInfo(model_dtype_str="bfloat16", override_existing_receiver=False),
        client,
    )
    original_empty = torch.empty
    monkeypatch.setattr(
        torch,
        "empty",
        lambda *args, **kwargs: original_empty(*args, **{**kwargs, "device": "cpu"}),
    )
    monkeypatch.setattr(
        cuda_ipc_strategy,
        "reduce_tensor",
        lambda t: (_rebuild_cpu, [t.tolist(), None, None, None, None, None, 0]),
    )
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "ipc_collect", lambda: None)
    other_metadata = {
        "names": ["expert.1"],
        "dtype_names": ["uint8"],
        "shapes": [[2]],
        "sizes": [2],
    }

    def gather(output, local):
        output[:] = [
            local,
            (
                "gpu-1",
                (_rebuild_cpu, [[7, 9], None, None, None, None, None, 99]),
                other_metadata,
            ),
        ]

    monkeypatch.setattr(torch.distributed, "all_gather_object", gather)
    chunk = WeightChunk(["expert.0"], ["uint8"], [[4]], [torch.arange(4, dtype=torch.uint8)])
    asyncio.run(
        sender._send_single_dtype_chunk_vllm_native(chunk=chunk, device=0, gpu_uuid="gpu-0", world_size=2, rank=0)
    )
    request = client.update_weights_ipc.await_args.args[0]
    assert request["metadata_by_gpu"]["gpu-1"] == other_metadata

    import vllm.config

    monkeypatch.setattr(vllm.config, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "current_device", lambda: receiver_rank)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(uuid=f"gpu-{receiver_rank}"),
    )
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    received = []
    worker = SimpleNamespace(
        _skyrl_weight_update_active=True,
        weight_transfer_engine=object(),
        device="cpu",
        vllm_config=None,
        model_runner=SimpleNamespace(model=object()),
        _skyrl_load_kernel_weights=received.extend,
    )
    NewInferenceWorkerWrap.update_weights_ipc(worker, request)
    assert [name for name, _ in received] == [f"expert.{receiver_rank}"]
    assert received[0][1].tolist() == ([0, 1, 2, 3] if receiver_rank == 0 else [7, 9])
