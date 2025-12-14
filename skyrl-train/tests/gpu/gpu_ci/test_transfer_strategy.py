"""
GPU integration tests for weight transfer strategies without involving the actual training workers and inference engines.

Tests the full execution flow:
    1. create_init_info: Extract config-derived args (master addr/port, dtype, etc.)
    2. create_sender/create_receiver: Initialize transfer components on both sides
    3. send_chunks/receive_weights: Transfer weight tensors between actors

Run with:
    uv run --isolated --extra dev pytest tests/gpu/gpu_ci/test_transfer_strategy.py -v
"""

import ray
import torch
import torch.distributed as dist
from unittest.mock import MagicMock

from skyrl_train.weight_sync import (
    WeightChunk,
    CudaIpcTransferStrategy,
    BroadcastTransferStrategy,
)
from skyrl_train.utils.utils import get_free_port


def make_mock_cfg(
    weight_sync_backend: str = "nccl",
    model_dtype: str = "torch.bfloat16",
    num_inference_engines: int = 1,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
    colocate_all: bool = True,
):
    """Create a mock config object."""
    cfg = MagicMock()
    cfg.generator.weight_sync_backend = weight_sync_backend
    cfg.generator.model_dtype = model_dtype
    cfg.generator.num_inference_engines = num_inference_engines
    cfg.generator.inference_engine_tensor_parallel_size = tensor_parallel_size
    cfg.generator.inference_engine_pipeline_parallel_size = pipeline_parallel_size
    cfg.generator.inference_engine_data_parallel_size = data_parallel_size
    cfg.trainer.placement.colocate_all = colocate_all
    return cfg


@ray.remote(num_gpus=1)
class SenderActor:
    """Generic sender actor for transfer strategies."""

    def create_init_info(self, strategy_cls, cfg):
        """Create init info using the strategy."""
        return strategy_cls.create_init_info(cfg)

    async def send_weights(
        self, strategy_cls, init_info, names: list, shapes: list, receiver_handle, send_individually: bool = False
    ):
        """Create sender and send weights."""
        dtype_str = init_info.model_dtype_str
        dtype = getattr(torch, dtype_str.replace("torch.", ""))
        tensors = [torch.randn(shape, device="cuda", dtype=dtype) for shape in shapes]

        class MockInferenceClient:
            def __init__(self, receiver_handle):
                self.receiver_handle = receiver_handle

            async def update_named_weights(self, request):
                ray.get(self.receiver_handle.receive_weights.remote(request))

        mock_client = MockInferenceClient(receiver_handle)
        sender = strategy_cls.create_sender(init_info, mock_client)

        if send_individually:
            for name, tensor, shape in zip(names, tensors, shapes):
                chunk = WeightChunk(names=[name], dtypes=[dtype_str], shapes=[shape], tensors=[tensor])
                await sender.send_chunks([chunk])
        else:
            chunk = WeightChunk(names=names, dtypes=[dtype_str] * len(names), shapes=shapes, tensors=tensors)
            await sender.send_chunks([chunk])

        return [t.cpu() for t in tensors]


@ray.remote(num_gpus=1)
class ReceiverActor:
    """Generic receiver actor for transfer strategies."""

    def __init__(self, strategy_cls, init_info, init_process_group: bool = False):
        if init_process_group:
            # Initialize default process group so torch.distributed.get_rank() works.
            # In production, this is the inference engine's model parallel group.
            port = get_free_port()
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://localhost:{port}",
                world_size=1,
                rank=0,
            )
        self.receiver = strategy_cls.create_receiver(init_info)
        self.received_weights = []

    def receive_weights(self, request):
        """Receive weights using the pre-created receiver."""
        received = list(self.receiver.receive_weights(request))
        self.received_weights.extend([(name, tensor.cpu()) for name, tensor in received])

    def get_received_weights(self):
        return self.received_weights


def _run_weight_sync_e2e(strategy_cls, cfg, init_process_group: bool, send_individually: bool):
    """Run end-to-end weight sync test for a given strategy."""
    sender = SenderActor.remote()
    init_info = ray.get(sender.create_init_info.remote(strategy_cls, cfg))

    receiver = ReceiverActor.remote(strategy_cls, init_info, init_process_group=init_process_group)

    names = ["layer1.weight", "layer1.bias", "layer2.weight"]
    shapes = [[32, 64], [64], [16, 16]]
    src_tensors = ray.get(
        sender.send_weights.remote(
            strategy_cls, init_info, names, shapes, receiver, send_individually=send_individually
        )
    )

    received = ray.get(receiver.get_received_weights.remote())

    assert len(received) == len(names)
    for i, (name, tensor) in enumerate(received):
        assert name == names[i]
        assert tensor.shape == tuple(shapes[i])
        assert torch.allclose(tensor, src_tensors[i])


class TestCudaIpcTransferStrategy:
    """Integration tests for CUDA IPC transfer strategy."""

    def test_weight_sync_e2e(self, ray_init_fixture):
        """Test CUDA IPC strategy end-to-end."""
        cfg = make_mock_cfg(model_dtype="torch.bfloat16", colocate_all=True)
        _run_weight_sync_e2e(CudaIpcTransferStrategy, cfg, init_process_group=False, send_individually=False)


class TestBroadcastTransferStrategy:
    """Integration tests for Broadcast transfer strategy."""

    def test_weight_sync_e2e(self, ray_init_fixture):
        """Test Broadcast strategy end-to-end."""
        cfg = make_mock_cfg(weight_sync_backend="nccl", model_dtype="torch.bfloat16", colocate_all=False)
        _run_weight_sync_e2e(BroadcastTransferStrategy, cfg, init_process_group=True, send_individually=True)
