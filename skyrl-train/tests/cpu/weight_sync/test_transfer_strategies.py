import pytest

from skyrl_train.config import SkyRLConfig
from skyrl_train.weight_sync import (
    get_transfer_strategy_cls,
    BroadcastTransferStrategy,
    CudaIpcTransferStrategy,
    BroadcastInitInfo,
    CudaIpcInitInfo,
    BroadcastWeightUpdateRequest,
    CudaIpcWeightUpdateRequest,
    LoraLoadRequest,
)


class TestGetTransferStrategyCls:
    """Tests for get_transfer_strategy_cls function."""

    def _make_cfg(self, weight_sync_backend: str, colocate_all: bool):
        """Create a config object."""
        cfg = SkyRLConfig()
        cfg.generator.weight_sync_backend = weight_sync_backend
        cfg.trainer.placement.colocate_all = colocate_all
        return cfg

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
        cfg = self._make_cfg(weight_sync_backend=backend, colocate_all=colocate_all)
        assert get_transfer_strategy_cls(cfg) is expected_strategy


class TestCreateInitInfo:
    """Tests for create_init_info static methods."""

    def _make_cfg(
        self,
        weight_sync_backend: str = "nccl",
        model_dtype: str = "torch.bfloat16",
        num_inference_engines: int = 1,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
        override_existing_update_group: str = "enable",
    ):
        """Create a config object for create_init_info."""
        cfg = SkyRLConfig()
        cfg.generator.weight_sync_backend = weight_sync_backend
        cfg.generator.model_dtype = model_dtype
        cfg.generator.num_inference_engines = num_inference_engines
        cfg.generator.inference_engine_tensor_parallel_size = tensor_parallel_size
        cfg.generator.inference_engine_pipeline_parallel_size = pipeline_parallel_size
        cfg.generator.inference_engine_data_parallel_size = data_parallel_size
        cfg.generator.override_existing_update_group = override_existing_update_group
        return cfg

    def test_cuda_ipc_create_init_info(self):
        """CudaIpcTransferStrategy.create_init_info should create CudaIpcInitInfo with model_dtype_str."""
        cfg = self._make_cfg(model_dtype="torch.float32")
        init_info = CudaIpcTransferStrategy.create_init_info(cfg)

        assert isinstance(init_info, CudaIpcInitInfo)
        assert init_info.model_dtype_str == "torch.float32"

    def test_broadcast_create_init_info(self, monkeypatch):
        """BroadcastTransferStrategy.create_init_info should create BroadcastInitInfo with correct fields."""
        # Mock ray to avoid actual network operations
        import skyrl_train.weight_sync.broadcast_strategy as broadcast_module

        monkeypatch.setattr(broadcast_module.ray._private.services, "get_node_ip_address", lambda: "192.168.1.1")

        cfg = self._make_cfg(
            weight_sync_backend="gloo",
            model_dtype="torch.bfloat16",
            num_inference_engines=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            override_existing_update_group="enable",
        )
        init_info = BroadcastTransferStrategy.create_init_info(cfg)

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
        import skyrl_train.weight_sync.broadcast_strategy as broadcast_module

        monkeypatch.setattr(broadcast_module.ray._private.services, "get_node_ip_address", lambda: "192.168.1.1")

        cfg = self._make_cfg(override_existing_update_group="disable")
        init_info = BroadcastTransferStrategy.create_init_info(cfg)

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
        from skyrl_train.weight_sync.cuda_ipc_strategy import _IPC_REQUEST_END_MARKER

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
