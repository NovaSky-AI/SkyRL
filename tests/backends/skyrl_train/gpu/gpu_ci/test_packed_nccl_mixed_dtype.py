"""Two-GPU NCCL coverage for SkyRL's serialized FP8 packed wire stream.

The CPU boundary tests assert that SkyRL hands its expanded stream to vLLM. This
test runs vLLM's actual packed NCCL producer and consumer on separate Ray GPU
workers, so the byte wire format is covered too. The tensor geometry matches the
Qwen dense projection path: an FP8 tensor whose byte length leaves the following
FP32 scale at a valid offset, followed by an ordinary BF16 tensor.
"""

from __future__ import annotations

import socket

import pytest
import ray


def _serialized_pairs(device):
    """Build the deterministic FP8 / FP32 / BF16 source stream on ``device``."""

    import torch

    from skyrl.backends.skyrl_train.weight_sync.fp8 import SerializedFp8Config
    from skyrl.backends.skyrl_train.weight_sync.fp8.models import QWEN35_FP8_SPEC
    from skyrl.backends.skyrl_train.weight_sync.sources import SerializedFp8WeightSource

    class DenseSource:
        def __iter__(self):
            # 128 x 128 gives one 128 x 128 FP8 block. Its 16,384-byte payload
            # leaves the following FP32 scale at a 4-byte-aligned offset.
            projection = torch.arange(128 * 128, device=device, dtype=torch.float32).reshape(128, 128)
            yield "model.layers.0.mlp.down_proj.weight", (projection / 512).to(torch.bfloat16)
            yield "model.layers.0.input_layernorm.weight", torch.linspace(
                -1,
                1,
                128,
                device=device,
                dtype=torch.bfloat16,
            )

    source = SerializedFp8WeightSource(DenseSource(), SerializedFp8Config(spec=QWEN35_FP8_SPEC))
    return list(source)


@ray.remote(num_gpus=1)
class _PackedNcclProducer:
    """Rank-zero producer; an actor keeps the selected rendezvous node stable."""

    def endpoint(self):
        from ray.util import get_node_ip_address

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return get_node_ip_address(), sock.getsockname()[1]

    def send(self, host, port):
        import torch
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.utils import StatelessProcessGroup
        from vllm.distributed.weight_transfer.packed_tensor import (
            packed_nccl_broadcast_producer,
        )

        torch.cuda.set_device(0)
        device = torch.device("cuda:0")
        group = StatelessProcessGroup.create(host=host, port=port, rank=0, world_size=2)
        communicator = PyNcclCommunicator(group, device=device)
        pairs = _serialized_pairs(device)
        try:
            packed_nccl_broadcast_producer(
                iterator=iter(pairs),
                group=communicator,
                src=0,
                post_iter_func=lambda item: item[1],
                buffer_size_bytes=64 * 1024,
                num_buffers=2,
            )
            torch.cuda.synchronize()
            return [(name, str(tensor.dtype), list(tensor.shape)) for name, tensor in pairs]
        finally:
            communicator.destroy()


@ray.remote(num_gpus=1)
def _consume_packed_serialized_stream(host, port):
    import torch
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.distributed.weight_transfer.packed_tensor import (
        packed_nccl_broadcast_consumer,
    )

    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    group = StatelessProcessGroup.create(host=host, port=port, rank=1, world_size=2)
    communicator = PyNcclCommunicator(group, device=device)
    expected = dict(_serialized_pairs(device))
    received = {}
    try:
        packed_nccl_broadcast_consumer(
            iterator=iter((name, (tuple(tensor.shape), tensor.dtype)) for name, tensor in expected.items()),
            group=communicator,
            src=0,
            post_unpack_func=lambda tensors: received.update(tensors),
            buffer_size_bytes=64 * 1024,
            num_buffers=2,
            device=device,
        )
        torch.cuda.synchronize()
        assert list(received) == list(expected)
        for name, expected_tensor in expected.items():
            actual = received[name]
            assert actual.dtype is expected_tensor.dtype
            assert actual.shape == expected_tensor.shape
            assert torch.equal(actual, expected_tensor), name
        return [(name, str(tensor.dtype), list(tensor.shape)) for name, tensor in received.items()]
    finally:
        communicator.destroy()


def test_serialized_fp8_mixed_dtypes_roundtrip_over_packed_nccl(ray_init_fixture):
    """The actual two-GPU producer/consumer roundtrip preserves every wire dtype."""

    if ray.cluster_resources().get("GPU", 0) < 2:
        pytest.skip("requires two GPUs")

    producer = _PackedNcclProducer.remote()
    host, port = ray.get(producer.endpoint.remote())
    consumer_result = _consume_packed_serialized_stream.remote(host, port)
    producer_result = producer.send.remote(host, port)
    sent, received = ray.get([producer_result, consumer_result])

    assert received == sent
    assert [dtype for _, dtype, _ in received] == [
        "torch.float8_e4m3fn",
        "torch.float32",
        "torch.bfloat16",
    ]
