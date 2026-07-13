import asyncio

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync import (
    BroadcastInitInfo,
    BroadcastTransferStrategy,
    BroadcastWeightUpdateRequest,
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    CudaIpcWeightUpdateRequest,
    LoraLoadRequest,
    get_transfer_strategy_cls,
)
from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.broadcast_strategy import (
    BroadcastWeightTransferSender,
)
from skyrl.train.config import InferenceEngineConfig


class TestGetTransferStrategyCls:
    """Tests for get_transfer_strategy_cls function."""

    @pytest.mark.parametrize(
        "backend,colocate_all,expected_strategy",
        [
            ("nccl", True, CudaIpcTransferStrategy),
            ("nccl", False, BroadcastTransferStrategy),
            ("gloo", True, BroadcastTransferStrategy),
            ("gloo", False, BroadcastTransferStrategy),
        ],
    )
    def test_returns_correct_strategy(self, backend, colocate_all, expected_strategy):
        """Should return correct strategy based on backend and colocate_all."""
        assert get_transfer_strategy_cls(backend, colocate_all) is expected_strategy


class TestCreateInitInfo:
    """Tests for create_init_info static methods."""

    def _make_ie_cfg(
        self,
        weight_sync_backend: str = "nccl",
        model_dtype: str = "torch.bfloat16",
        num_engines: int = 1,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
        run_engines_locally: bool = True,
    ):
        """Create an InferenceEngineConfig for create_init_info."""
        return InferenceEngineConfig(
            weight_sync_backend=weight_sync_backend,
            model_dtype=model_dtype,
            num_engines=num_engines,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            run_engines_locally=run_engines_locally,
        )

    def test_cuda_ipc_create_init_info(self):
        """CudaIpcTransferStrategy.create_init_info preserves model dtype metadata."""
        ie_cfg = self._make_ie_cfg(model_dtype="torch.float32")
        init_info = CudaIpcTransferStrategy.create_init_info(ie_cfg)

        assert isinstance(init_info, CudaIpcInitInfo)
        assert init_info.model_dtype_str == "torch.float32"

    def test_broadcast_create_init_info(self, monkeypatch):
        """BroadcastTransferStrategy.create_init_info should create BroadcastInitInfo with correct fields."""
        # Mock ray to avoid actual network operations
        import skyrl.backends.skyrl_train.weight_sync.broadcast_strategy as broadcast_module

        monkeypatch.setattr(broadcast_module.ray._private.services, "get_node_ip_address", lambda: "192.168.1.1")

        ie_cfg = self._make_ie_cfg(
            weight_sync_backend="gloo",
            model_dtype="torch.bfloat16",
            num_engines=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            run_engines_locally=False,
        )
        init_info = BroadcastTransferStrategy.create_init_info(ie_cfg, inference_world_size=4)

        assert isinstance(init_info, BroadcastInitInfo)
        assert init_info.master_addr == "192.168.1.1"
        assert isinstance(init_info.master_port, int)
        assert init_info.rank_offset == 1
        # world_size = inference_world_size + 1 = 4 + 1 = 5
        assert init_info.world_size == 5
        assert init_info.override_existing_receiver is True

    def test_broadcast_create_init_info_override_existing_receiver_disabled_for_local_engines(self, monkeypatch):
        """BroadcastTransferStrategy.create_init_info should set override_existing_receiver=False for local engines."""
        import skyrl.backends.skyrl_train.weight_sync.broadcast_strategy as broadcast_module

        monkeypatch.setattr(broadcast_module.ray._private.services, "get_node_ip_address", lambda: "192.168.1.1")

        ie_cfg = self._make_ie_cfg(run_engines_locally=True)
        init_info = BroadcastTransferStrategy.create_init_info(ie_cfg, inference_world_size=1)

        assert init_info.override_existing_receiver is False


class TestBroadcastWeightUpdateRequest:
    """Tests for BroadcastWeightUpdateRequest."""

    def test_len(self):
        """__len__ should return number of weights."""
        request = BroadcastWeightUpdateRequest(
            names=["layer1.weight", "layer2.weight"],
            dtypes=["bfloat16", "bfloat16"],
            shapes=[[4096, 4096], [1024]],
        )
        assert len(request) == 2

    def test_mismatched_lengths_raises(self):
        """Mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            BroadcastWeightUpdateRequest(
                names=["layer1.weight", "layer2.weight"],
                dtypes=["bfloat16"],
                shapes=[[4096, 4096]],
            )


def test_broadcast_sender_preserves_mixed_dtype_logical_chunk(monkeypatch):
    """vLLM's byte-packed NCCL path accepts one mixed-dtype logical update."""
    import skyrl.backends.skyrl_train.weight_sync.broadcast_strategy as broadcast_module

    class FakeInferenceClient:
        def __init__(self):
            self.events = []

        async def start_weight_update(self, is_checkpoint_format):
            self.events.append(("start", is_checkpoint_format))

        async def finish_weight_update(self):
            self.events.append(("finish",))

    client = FakeInferenceClient()
    sender = BroadcastWeightTransferSender(
        init_info=BroadcastInitInfo(
            master_addr="127.0.0.1",
            master_port=12345,
            rank_offset=1,
            world_size=2,
            override_existing_receiver=False,
        ),
        model_update_group=object(),
        inference_client=client,
    )
    sent_chunks = []

    async def record_chunk(chunk):
        sent_chunks.append((list(chunk.names), [tensor.dtype for tensor in chunk.tensors]))

    monkeypatch.setattr(sender, "_send_chunk_vllm_native", record_chunk)
    monkeypatch.setattr(broadcast_module.torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(broadcast_module.torch.distributed, "barrier", lambda: None)

    mixed_chunk = WeightChunk(
        names=["w0", "scale", "w1", "norm"],
        dtypes=["ignored"] * 4,
        shapes=[[4], [1], [8], [2]],
        tensors=[
            torch.empty((4,), dtype=torch.float8_e4m3fn),
            torch.empty((1,), dtype=torch.float32),
            torch.empty((8,), dtype=torch.float8_e4m3fn),
            torch.empty((2,), dtype=torch.bfloat16),
        ],
    )

    asyncio.run(sender.send_chunks(iter([mixed_chunk]), derive_metadata_from_chunks=True))

    assert client.events == [("start", True), ("finish",)]
    assert sent_chunks == [
        (
            ["w0", "scale", "w1", "norm"],
            [torch.float8_e4m3fn, torch.float32, torch.float8_e4m3fn, torch.bfloat16],
        )
    ]


def test_broadcast_sender_retains_precomputed_metadata_path(monkeypatch):
    sender = BroadcastWeightTransferSender(
        init_info=BroadcastInitInfo(
            master_addr="127.0.0.1",
            master_port=12345,
            rank_offset=1,
            world_size=2,
            override_existing_receiver=False,
        ),
        model_update_group=object(),
        inference_client=object(),
    )
    calls = []

    async def record_batched(chunks, weight_metadata):
        calls.append((list(chunks), weight_metadata))

    monkeypatch.setattr(sender, "_send_chunks_vllm_native", record_batched)
    metadata = {"names": ["w"], "dtype_names": ["bfloat16"], "shapes": [[1]]}

    asyncio.run(sender.send_chunks(iter([]), weight_metadata=metadata))

    assert calls == [([], metadata)]


class TestCudaIpcWeightUpdateRequest:
    """Tests for CudaIpcWeightUpdateRequest."""

    def test_serialize_roundtrip(self):
        """Serialization/deserialization roundtrip preserves data."""
        request = CudaIpcWeightUpdateRequest(
            names=["model.layer.weight"],
            dtypes=["bfloat16"],
            shapes=[[4096, 4096]],
            sizes=[4096 * 4096],
            ipc_handles={"gpu-uuid": "test_handle"},
        )

        data = request.serialize()
        result = CudaIpcWeightUpdateRequest.deserialize(data)

        assert result.names == request.names
        assert result.dtypes == request.dtypes
        assert result.shapes == request.shapes
        assert result.sizes == request.sizes
        assert result.ipc_handles == request.ipc_handles

    def test_serialize_roundtrip_multiple_weights(self):
        """Roundtrip with multiple weights."""
        request = CudaIpcWeightUpdateRequest(
            names=["layer1.weight", "layer2.weight", "layer3.bias"],
            dtypes=["bfloat16", "bfloat16", "bfloat16"],
            shapes=[[4096, 4096], [4096, 1024], [1024]],
            sizes=[4096 * 4096, 4096 * 1024, 1024],
            ipc_handles={"gpu-0": "handle1"},
        )

        data = request.serialize()
        result = CudaIpcWeightUpdateRequest.deserialize(data)

        assert result.names == request.names
        assert result.dtypes == request.dtypes
        assert result.shapes == request.shapes
        assert result.sizes == request.sizes
        assert result.ipc_handles == request.ipc_handles

    def test_deserialize_missing_end_marker(self):
        """Missing end marker raises ValueError."""

        invalid_data = b"some_invalid_data"

        with pytest.raises(ValueError, match="End marker not found"):
            CudaIpcWeightUpdateRequest.deserialize(invalid_data)

    def test_deserialize_invalid_data(self):
        """Invalid base64/pickle data raises ValueError."""
        from skyrl.backends.skyrl_train.weight_sync.cuda_ipc_strategy import (
            _IPC_REQUEST_END_MARKER,
        )

        invalid_data = b"not_valid_base64!!!" + _IPC_REQUEST_END_MARKER

        with pytest.raises(ValueError, match="Failed to deserialize"):
            CudaIpcWeightUpdateRequest.deserialize(invalid_data)

    def test_serialize_aligned_to_4_bytes(self):
        """Serialized data is 4-byte aligned."""
        request = CudaIpcWeightUpdateRequest(
            names=["test"],
            dtypes=["bfloat16"],
            shapes=[[10]],
            sizes=[10],
            ipc_handles={},
        )
        data = request.serialize()

        assert len(data) % 4 == 0


class TestLoraLoadRequest:
    """Tests for LoraLoadRequest."""

    def test_lora_path(self):
        """lora_path should be stored correctly with empty defaults for base fields."""
        request = LoraLoadRequest(lora_path="/path/to/lora")
        assert request.lora_path == "/path/to/lora"
        assert request.names == []
        assert request.dtypes == []
        assert request.shapes == []
