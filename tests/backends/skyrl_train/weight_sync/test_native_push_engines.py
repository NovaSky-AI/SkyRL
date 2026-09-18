"""SkyRL coverage at the native vLLM push-engine boundary.

The old ``WeightChunk`` senders owned chunk metadata and the start/update/finish
lifecycle.  The native path instead gives a ``WeightSource`` to vLLM's NCCL or
IPC trainer engine.  These tests pin SkyRL's side of that boundary without a
GPU: vLLM's CUDA packers are replaced with recorders while the real trainer
engines build their update payloads and drive the control-plane client.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm", reason="native trainer engines require the vLLM wheel")

pytestmark = pytest.mark.vllm

from vllm.distributed.weight_transfer.base import ParamMeta, WeightSource  # noqa: E402
from vllm.distributed.weight_transfer.packed_tensor import PackedIpcChunk  # noqa: E402

from skyrl.backends.skyrl_train.weight_sync.fp8 import SerializedFp8Config  # noqa: E402
from skyrl.backends.skyrl_train.weight_sync.fp8.models import QWEN35_FP8_SPEC  # noqa: E402
from skyrl.backends.skyrl_train.weight_sync.sources import SerializedFp8WeightSource  # noqa: E402
from skyrl.backends.skyrl_train.weight_sync.weight_senders import (  # noqa: E402
    get_skyrl_ipc_trainer,
    get_skyrl_nccl_trainer,
)


class _RecordingClient:
    """Synchronous vLLM client double that records the native engine contract."""

    def __init__(self):
        self.events: list[tuple[str, object | None]] = []
        self.update_infos: list[dict] = []

    def init_weight_transfer_engine(self, init_info):
        self.events.append(("init", init_info))

    def start_weight_update(self):
        self.events.append(("start", None))

    def update_weights(self, update_info):
        self.update_infos.append(update_info)
        self.events.append(("update", update_info))

    def finish_weight_update(self):
        self.events.append(("finish", None))


class _StaticWeightSource(WeightSource):
    """A normal source with cheap, precomputed metadata."""

    def __init__(self, pairs: list[tuple[str, torch.Tensor]]):
        self._pairs = pairs
        self._metadata = [ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in pairs]
        self.metadata_calls = 0
        self.iterations = 0

    def metadata(self) -> list[ParamMeta]:
        self.metadata_calls += 1
        return self._metadata

    def __iter__(self):
        self.iterations += 1
        yield from self._pairs


def _serialized_fp8_source() -> SerializedFp8WeightSource:
    """A real FP8-expanded stream: FP8 weight, FP32 scale, then BF16 weight."""

    class DenseSource:
        def __iter__(self):
            yield (
                "model.layers.0.mlp.down_proj.weight",
                torch.ones(128, 128, dtype=torch.bfloat16),
            )
            yield (
                "model.layers.0.input_layernorm.weight",
                torch.ones(128, dtype=torch.bfloat16),
            )

    return SerializedFp8WeightSource(DenseSource(), SerializedFp8Config(spec=QWEN35_FP8_SPEC))


def _stream_signature(source: WeightSource) -> list[tuple[str, torch.dtype, tuple[int, ...]]]:
    return [(meta.name, meta.dtype, meta.shape) for meta in source.metadata()]


def _assert_update_matches(source: WeightSource, update_info: dict) -> None:
    expected = _stream_signature(source)
    assert list(zip(update_info["names"], update_info["dtype_names"], update_info["shapes"])) == [
        (name, str(dtype).split(".")[-1], list(shape)) for name, dtype, shape in expected
    ]


def test_native_nccl_trainer_preserves_serialized_fp8_mixed_stream(monkeypatch):
    """Replacement for the old mixed ``WeightChunk`` broadcast test.

    ``SerializedFp8WeightSource`` is passed directly to vLLM's trainer engine.
    The engine, rather than SkyRL, creates the update metadata and hands the
    same ordered FP8/FP32/BF16 stream to the packed NCCL producer.
    """

    import vllm.distributed.weight_transfer.nccl_engine as nccl_engine

    source = _serialized_fp8_source()
    client = _RecordingClient()
    observed: dict[str, object] = {}

    def record_packed_producer(*, iterator, group, src, post_iter_func, buffer_size_bytes, num_buffers):
        observed["group"] = group
        observed["src"] = src
        observed["buffer_size_bytes"] = buffer_size_bytes
        observed["num_buffers"] = num_buffers
        observed["stream"] = [
            (name, post_iter_func((name, tensor))) for name, tensor in iterator
        ]

    monkeypatch.setattr(nccl_engine, "packed_nccl_broadcast_producer", record_packed_producer)

    _, engine_cls = get_skyrl_nccl_trainer()
    engine = engine_cls(client=client, source=source, packed=True)
    engine.model_update_group = object()
    engine.send_weights()

    _assert_update_matches(source, client.update_infos[0])
    assert "packed" not in client.update_infos[0], "packed is agreed during engine initialization"
    assert [tensor.dtype for _, tensor in observed["stream"]] == [
        torch.float8_e4m3fn,
        torch.float32,
        torch.bfloat16,
    ]
    assert [name for name, _ in observed["stream"]] == client.update_infos[0]["names"]
    assert [event for event, _ in client.events] == ["start", "update", "finish"]


def test_native_nccl_trainer_agrees_packing_during_initialization(monkeypatch):
    """Packing is a NCCL init-handshake property, never an update field.

    This is the native equivalent of the old sender test that checked that the
    broadcast request used the packing configuration supplied at initialization.
    """

    import vllm.distributed.weight_transfer.nccl_engine as nccl_engine

    source = _StaticWeightSource(
        [("model.norm.weight", torch.ones(4, dtype=torch.bfloat16))]
    )
    client = _RecordingClient()

    monkeypatch.setattr(nccl_engine, "open_trainer_endpoint", lambda _init_info: object())

    init_info_cls, engine_cls = get_skyrl_nccl_trainer()
    engine = engine_cls.trainer_init(
        init_info_cls(
            master_address="127.0.0.1",
            master_port=12345,
            world_size=5,
            rank=0,
            packed=True,
            packed_buffer_size_bytes=1024,
            packed_num_buffers=3,
        ),
        client=client,
        source=source,
    )

    assert client.events == [
        (
            "init",
            {
                "master_address": "127.0.0.1",
                "master_port": 12345,
                "rank_offset": 1,
                "world_size": 5,
                "packed": True,
                "packed_buffer_size_bytes": 1024,
                "packed_num_buffers": 3,
            },
        )
    ]
    assert engine.packed is True
    assert engine.packed_buffer_size_bytes == 1024
    assert engine.packed_num_buffers == 3


def test_native_nccl_trainer_uses_precomputed_weight_source_metadata(monkeypatch):
    """Replacement for the old precomputed-metadata sender branch.

    There is no longer a SkyRL ``send_chunks`` branch.  A normal source declares
    metadata and vLLM derives the update payload from it before iterating the
    source into its packed producer.
    """

    import vllm.distributed.weight_transfer.nccl_engine as nccl_engine

    source = _StaticWeightSource(
        [
            ("model.layers.0.self_attn.q_proj.weight", torch.ones(4, 4, dtype=torch.bfloat16)),
            ("model.norm.weight", torch.ones(4, dtype=torch.bfloat16)),
        ]
    )
    client = _RecordingClient()
    observed: list[tuple[str, torch.Tensor]] = []

    def record_packed_producer(*, iterator, **_kwargs):
        observed.extend(iterator)

    monkeypatch.setattr(nccl_engine, "packed_nccl_broadcast_producer", record_packed_producer)

    _, engine_cls = get_skyrl_nccl_trainer()
    engine = engine_cls(client=client, source=source, packed=True)
    engine.model_update_group = object()
    engine.send_weights()

    _assert_update_matches(source, client.update_infos[0])
    assert source.metadata_calls == 2  # Native send plus the assertion above.
    assert source.iterations == 1
    assert [name for name, _ in observed] == client.update_infos[0]["names"]
    assert [event for event, _ in client.events] == ["start", "update", "finish"]


def test_native_ipc_trainer_preserves_serialized_fp8_mixed_stream(monkeypatch):
    """Exercise the new native packed-IPC handoff without a CUDA allocation."""

    import vllm.distributed.weight_transfer.ipc_engine as ipc_engine

    source = _serialized_fp8_source()
    client = _RecordingClient()
    observed: dict[str, object] = {}

    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _index: SimpleNamespace(uuid="GPU-0"))
    monkeypatch.setattr(ipc_engine.IPCTrainerWeightTransferEngine, "_post_send_sync", staticmethod(lambda: None))

    def record_packed_ipc_producer(*, iterator, gpu_uuid, post_iter_func, buffer_size_bytes):
        pairs = [(name, post_iter_func((name, tensor))) for name, tensor in iterator]
        observed["stream"] = pairs
        observed["buffer_size_bytes"] = buffer_size_bytes
        yield PackedIpcChunk(
            names=[name for name, _ in pairs],
            shapes=[list(tensor.shape) for _, tensor in pairs],
            dtype_names=[str(tensor.dtype).split(".")[-1] for _, tensor in pairs],
            tensor_sizes=[tensor.nbytes for _, tensor in pairs],
            ipc_handle={gpu_uuid: ("unused-in-cpu-test",)},
        )

    monkeypatch.setattr(ipc_engine, "packed_ipc_producer", record_packed_ipc_producer)

    init_info_cls, engine_cls = get_skyrl_ipc_trainer()
    engine = engine_cls.trainer_init(
        init_info_cls(rank=0, packed=True, packed_buffer_size_bytes=1024),
        client=client,
        source=source,
    )
    engine.send_weights()

    _assert_update_matches(source, client.update_infos[0])
    assert client.events[0] == ("init", {"packed": True})
    assert [event for event, _ in client.events] == ["init", "start", "update", "finish"]
    assert [tensor.dtype for _, tensor in observed["stream"]] == [
        torch.float8_e4m3fn,
        torch.float32,
        torch.bfloat16,
    ]
    assert observed["buffer_size_bytes"] == 1024
    assert client.update_infos[0]["tensor_sizes"] == [tensor.nbytes for _, tensor in observed["stream"]]
