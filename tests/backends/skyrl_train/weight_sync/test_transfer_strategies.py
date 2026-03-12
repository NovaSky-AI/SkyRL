import pytest
import torch

from skyrl.train.config import InferenceEngineConfig
from skyrl.backends.skyrl_train.weight_sync import (
    get_transfer_strategy_cls,
    BroadcastTransferStrategy,
    CudaIpcTransferStrategy,
    RdtTransferStrategy,
    RdtWeightTransferReceiver,
    BroadcastInitInfo,
    CudaIpcInitInfo,
    RdtInitInfo,
    BroadcastWeightUpdateRequest,
    CudaIpcWeightUpdateRequest,
    LoraLoadRequest,
)


class TestGetTransferStrategyCls:
    """Tests for get_transfer_strategy_cls function."""

    @pytest.mark.parametrize(
        "backend,colocate_all,expected_strategy",
        [
            ("nccl", True, CudaIpcTransferStrategy),
            ("nccl", False, BroadcastTransferStrategy),
            ("gloo", True, BroadcastTransferStrategy),
            ("gloo", False, BroadcastTransferStrategy),
            ("rdt", True, RdtTransferStrategy),
            ("rdt", False, RdtTransferStrategy),
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
        override_existing_update_group: str = "enable",
    ):
        """Create an InferenceEngineConfig for create_init_info."""
        return InferenceEngineConfig(
            weight_sync_backend=weight_sync_backend,
            model_dtype=model_dtype,
            num_engines=num_engines,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            override_existing_update_group=override_existing_update_group,
        )

    def test_cuda_ipc_create_init_info(self):
        """CudaIpcTransferStrategy.create_init_info should create CudaIpcInitInfo with model_dtype_str."""
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
            override_existing_update_group="enable",
        )
        init_info = BroadcastTransferStrategy.create_init_info(ie_cfg)

        assert isinstance(init_info, BroadcastInitInfo)
        assert init_info.master_addr == "192.168.1.1"
        assert isinstance(init_info.master_port, int)
        assert init_info.rank_offset == 1
        # world_size = num_engines * tp * pp * dp + 1 = 2 * 2 * 1 * 1 + 1 = 5
        assert init_info.world_size == 5
        assert init_info.group_name == "skyrl"
        assert init_info.backend == "gloo"
        assert init_info.model_dtype_str == "torch.bfloat16"
        assert init_info.override_existing_receiver is True

    def test_broadcast_create_init_info_override_existing_receiver_disabled(self, monkeypatch):
        """BroadcastTransferStrategy.create_init_info should set override_existing_receiver=False when config is 'disable'."""
        import skyrl.backends.skyrl_train.weight_sync.broadcast_strategy as broadcast_module

        monkeypatch.setattr(broadcast_module.ray._private.services, "get_node_ip_address", lambda: "192.168.1.1")

        ie_cfg = self._make_ie_cfg(override_existing_update_group="disable")
        init_info = BroadcastTransferStrategy.create_init_info(ie_cfg)

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
        from skyrl.backends.skyrl_train.weight_sync.cuda_ipc_strategy import _IPC_REQUEST_END_MARKER

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


class TestRdtCreateInitInfo:
    """Tests for RdtTransferStrategy.create_init_info."""

    def _make_ie_cfg(self, **kwargs):
        defaults = {
            "weight_sync_backend": "rdt",
            "model_dtype": "torch.bfloat16",
            "num_engines": 1,
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "data_parallel_size": 1,
            "override_existing_update_group": "enable",
        }
        defaults.update(kwargs)
        return InferenceEngineConfig(**defaults)

    def test_create_init_info(self):
        """RdtTransferStrategy.create_init_info should create RdtInitInfo with model_dtype_str."""
        ie_cfg = self._make_ie_cfg(model_dtype="torch.float32")
        init_info = RdtTransferStrategy.create_init_info(ie_cfg)

        assert isinstance(init_info, RdtInitInfo)
        assert init_info.model_dtype_str == "torch.float32"
        assert init_info.override_existing_receiver is True

    def test_create_init_info_override_disabled(self):
        """RdtTransferStrategy.create_init_info should set override_existing_receiver=False when config is 'disable'."""
        ie_cfg = self._make_ie_cfg(override_existing_update_group="disable")
        init_info = RdtTransferStrategy.create_init_info(ie_cfg)

        assert init_info.override_existing_receiver is False

    def test_create_init_info_rejects_tp_gt_1(self):
        """RdtTransferStrategy.create_init_info should raise ValueError when TP > 1."""
        ie_cfg = self._make_ie_cfg(tensor_parallel_size=2)
        with pytest.raises(ValueError, match="TP=PP=1"):
            RdtTransferStrategy.create_init_info(ie_cfg)

    def test_create_init_info_rejects_pp_gt_1(self):
        """RdtTransferStrategy.create_init_info should raise ValueError when PP > 1."""
        ie_cfg = self._make_ie_cfg(pipeline_parallel_size=2)
        with pytest.raises(ValueError, match="TP=PP=1"):
            RdtTransferStrategy.create_init_info(ie_cfg)

    def test_strategy_type_roundtrip(self):
        """RdtInitInfo.strategy_type() should return RdtTransferStrategy."""
        ie_cfg = self._make_ie_cfg()
        init_info = RdtTransferStrategy.create_init_info(ie_cfg)
        assert init_info.strategy_type() is RdtTransferStrategy


class TestRdtReceiverUnpacking:
    """CPU-only tests for RdtWeightTransferReceiver.receive_weights unpacking logic."""

    def _pack_tensors(self, tensors, dtype=torch.bfloat16):
        """Pack a list of tensors into a contiguous 1D buffer (mimics sender logic)."""
        total_numel = sum(t.numel() for t in tensors)
        packed = torch.empty(total_numel, dtype=dtype)
        offset = 0
        for t in tensors:
            size = t.numel()
            packed[offset : offset + size].copy_(t.detach().view(-1))
            offset += size
        return packed

    def test_basic_unpacking(self):
        """Receiver should correctly unpack multiple tensors from a packed buffer."""
        t1 = torch.randn(32, 64, dtype=torch.bfloat16)
        t2 = torch.randn(64, dtype=torch.bfloat16)
        t3 = torch.randn(16, 16, dtype=torch.bfloat16)

        packed = self._pack_tensors([t1, t2, t3])
        metadata = {
            "names": ["layer1.weight", "layer1.bias", "layer2.weight"],
            "dtypes": ["bfloat16", "bfloat16", "bfloat16"],
            "shapes": [[32, 64], [64], [16, 16]],
            "sizes": [t1.numel(), t2.numel(), t3.numel()],
        }

        receiver = RdtWeightTransferReceiver(model_dtype=torch.bfloat16)
        results = list(receiver.receive_weights(packed, metadata))

        assert len(results) == 3
        assert results[0][0] == "layer1.weight"
        assert results[0][1].shape == (32, 64)
        assert torch.allclose(results[0][1], t1)
        assert results[1][0] == "layer1.bias"
        assert results[1][1].shape == (64,)
        assert torch.allclose(results[1][1], t2)
        assert results[2][0] == "layer2.weight"
        assert results[2][1].shape == (16, 16)
        assert torch.allclose(results[2][1], t3)

    def test_single_tensor(self):
        """Receiver should handle a single tensor."""
        t = torch.randn(8, 8, dtype=torch.bfloat16)
        packed = self._pack_tensors([t])
        metadata = {
            "names": ["single.weight"],
            "dtypes": ["bfloat16"],
            "shapes": [[8, 8]],
            "sizes": [t.numel()],
        }

        receiver = RdtWeightTransferReceiver(model_dtype=torch.bfloat16)
        results = list(receiver.receive_weights(packed, metadata))

        assert len(results) == 1
        assert results[0][0] == "single.weight"
        assert torch.allclose(results[0][1], t)

    def test_dtype_mismatch_raises(self):
        """Receiver should raise ValueError when metadata dtype doesn't match model dtype."""
        packed = torch.randn(10, dtype=torch.float32)
        metadata = {
            "names": ["w"],
            "dtypes": ["float32"],
            "shapes": [[10]],
            "sizes": [10],
        }

        receiver = RdtWeightTransferReceiver(model_dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="mismatch dtype"):
            list(receiver.receive_weights(packed, metadata))

    def test_mixed_dtypes_in_metadata_raises(self):
        """Receiver should raise ValueError when metadata contains mixed dtypes."""
        packed = torch.randn(20, dtype=torch.bfloat16)
        metadata = {
            "names": ["w1", "w2"],
            "dtypes": ["bfloat16", "float32"],
            "shapes": [[10], [10]],
            "sizes": [10, 10],
        }

        receiver = RdtWeightTransferReceiver(model_dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="packed weight update should have all tensors with the same dtype"):
            list(receiver.receive_weights(packed, metadata))

    def test_mismatched_sizes_names_raises(self):
        """Receiver should raise ValueError when sizes and names have different lengths."""
        packed = torch.randn(10, dtype=torch.bfloat16)
        metadata = {
            "names": ["w1", "w2"],
            "dtypes": ["bfloat16", "bfloat16"],
            "shapes": [[5], [5]],
            "sizes": [10],
        }

        receiver = RdtWeightTransferReceiver(model_dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="sizes must have the same length as names"):
            list(receiver.receive_weights(packed, metadata))

    def test_unpacked_tensors_are_views_of_packed_buffer(self):
        """Unpacked tensors should be views (not copies) of the packed buffer for zero-copy."""
        t1 = torch.randn(4, 4, dtype=torch.bfloat16)
        t2 = torch.randn(8, dtype=torch.bfloat16)
        packed = self._pack_tensors([t1, t2])
        metadata = {
            "names": ["w1", "w2"],
            "dtypes": ["bfloat16", "bfloat16"],
            "shapes": [[4, 4], [8]],
            "sizes": [16, 8],
        }

        receiver = RdtWeightTransferReceiver(model_dtype=torch.bfloat16)
        results = list(receiver.receive_weights(packed, metadata))

        assert results[0][1].data_ptr() == packed.data_ptr()
        assert results[1][1].data_ptr() == packed[16:].data_ptr()


class TestLoraLoadRequest:
    """Tests for LoraLoadRequest."""

    def test_lora_path(self):
        """lora_path should be stored correctly with empty defaults for base fields."""
        request = LoraLoadRequest(lora_path="/path/to/lora")
        assert request.lora_path == "/path/to/lora"
        assert request.names == []
        assert request.dtypes == []
        assert request.shapes == []
