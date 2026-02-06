import pytest

from skyrl_train.config import InferenceEngineConfig
from skyrl_train.weight_sync import (
    get_transfer_strategy_cls,
    BroadcastTransferStrategy,
    CudaIpcTransferStrategy,
    BroadcastInitInfo,
    CudaIpcInitInfo,
    BroadcastWeightUpdateRequest,
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
        import skyrl_train.weight_sync.broadcast_strategy as broadcast_module

        monkeypatch.setattr(broadcast_module.ray._private.services, "get_node_ip_address", lambda: "192.168.1.1")

        ie_cfg = self._make_ie_cfg(
            num_engines=2,
            tensor_parallel_size=4,
            pipeline_parallel_size=2,
            data_parallel_size=1,
            override_existing_update_group="enable",
        )
        init_info = BroadcastTransferStrategy.create_init_info(ie_cfg)

        assert isinstance(init_info, BroadcastInitInfo)
        assert init_info.master_addr == "192.168.1.1"
        assert init_info.world_size == 2 * 4 * 2 * 1 + 1  # num_engines * tp * pp * dp + 1 (trainer rank 0)
        assert init_info.override_existing_receiver is True

    def test_broadcast_create_init_info_override_existing_receiver_disabled(self, monkeypatch):
        """Verify override_existing_receiver can be disabled."""
        import skyrl_train.weight_sync.broadcast_strategy as broadcast_module

        monkeypatch.setattr(broadcast_module.ray._private.services, "get_node_ip_address", lambda: "10.0.0.1")

        ie_cfg = self._make_ie_cfg(override_existing_update_group="disable")
        init_info = BroadcastTransferStrategy.create_init_info(ie_cfg)

        assert init_info.override_existing_receiver is False


class TestWeightUpdateRequest:
    """Tests for weight update request creation."""

    def test_broadcast_weight_update_request(self):
        """BroadcastWeightUpdateRequest should store base fields."""
        req = BroadcastWeightUpdateRequest(
            names=["layer1.weight"],
            dtypes=["torch.float32"],
            shapes=[[768, 768]],
        )
        assert req.names == ["layer1.weight"]
        assert req.dtypes == ["torch.float32"]
        assert req.shapes == [[768, 768]]

    def test_lora_load_request(self):
        """LoraLoadRequest should store lora path."""
        req = LoraLoadRequest(
            lora_path="/path/to/lora",
        )
        assert req.lora_path == "/path/to/lora"
        assert req.names == []
        assert req.dtypes == []
        assert req.shapes == []
