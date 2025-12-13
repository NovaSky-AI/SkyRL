import pytest
from unittest.mock import MagicMock
from skyrl_train.weight_sync import (
    get_transfer_strategy_cls,
    BroadcastTransferStrategy,
    CudaIpcTransferStrategy,
    BroadcastInitInfo,
    CudaIpcInitInfo,
)


class TestGetTransferStrategyCls:
    """Tests for get_transfer_strategy_cls function."""

    def _make_cfg(self, weight_sync_backend: str, colocate_all: bool):
        """Create a mock config object."""
        cfg = MagicMock()
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
    ):
        """Create a mock config object for create_init_info."""
        cfg = MagicMock()
        cfg.generator.weight_sync_backend = weight_sync_backend
        cfg.generator.model_dtype = model_dtype
        cfg.generator.num_inference_engines = num_inference_engines
        cfg.generator.inference_engine_tensor_parallel_size = tensor_parallel_size
        cfg.generator.inference_engine_pipeline_parallel_size = pipeline_parallel_size
        cfg.generator.inference_engine_data_parallel_size = data_parallel_size
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
