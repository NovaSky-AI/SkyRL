import pytest

from skyrl.backends.skyrl_train.weight_sync import (
    RDT_TRAINER_ACTOR_NAME,
    BroadcastInitInfo,
    BroadcastTransferStrategy,
    BroadcastWeightUpdateRequest,
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    CudaIpcWeightUpdateRequest,
    LoraLoadRequest,
    ShardedRdtInitInfo,
    ShardedRdtTransferStrategy,
    get_transfer_strategy,
    get_transfer_strategy_cls,
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
            # sharded_rdt always selects the RDT strategy (non-colocated only is
            # enforced later in build_vllm_cli_args, not in strategy selection).
            ("sharded_rdt", True, ShardedRdtTransferStrategy),
            ("sharded_rdt", False, ShardedRdtTransferStrategy),
        ],
    )
    def test_returns_correct_strategy(self, backend, colocate_all, expected_strategy):
        """Should return correct strategy based on backend and colocate_all."""
        assert get_transfer_strategy_cls(backend, colocate_all) is expected_strategy

    @pytest.mark.parametrize(
        "backend,colocate_all,expected",
        [
            ("nccl", True, "ipc"),
            ("nccl", False, "nccl"),
            ("sharded_rdt", True, "sharded_rdt"),
            ("sharded_rdt", False, "sharded_rdt"),
        ],
    )
    def test_backend_string(self, backend, colocate_all, expected):
        """get_transfer_strategy maps to the vLLM WeightTransferConfig.backend string."""
        assert get_transfer_strategy(backend, colocate_all) == expected


class TestShardedRdtStrategy:
    """Tests for the sharded_rdt (NIXL pull) strategy — no GPU/vLLM needed."""

    def _make_ie_cfg(self) -> InferenceEngineConfig:
        return InferenceEngineConfig(
            weight_sync_backend="sharded_rdt",
            model_dtype="bfloat16",
            override_existing_update_group="enable",
        )

    def test_create_init_info(self):
        """create_init_info returns a ShardedRdtInitInfo with the trainer actor name."""
        init_info = ShardedRdtTransferStrategy.create_init_info(self._make_ie_cfg())
        assert isinstance(init_info, ShardedRdtInitInfo)
        assert init_info.trainer_actor_name == RDT_TRAINER_ACTOR_NAME
        assert init_info.produce_method_name == "rdt_produce_weights_batched"
        assert init_info.warmup_method_name == "rdt_warmup"
        assert init_info.override_existing_receiver is True
        # names/dtype_names/shapes are filled in later from the weight extractor.
        assert init_info.names == []
        assert init_info.strategy_type() is ShardedRdtTransferStrategy

    def test_to_api_payload(self):
        """to_api_payload matches the vLLM ShardedRDTWeightTransferInitInfo fields."""
        init_info = ShardedRdtTransferStrategy.create_init_info(self._make_ie_cfg())
        init_info.names = ["model.embed_tokens.weight", "model.layers.0.mlp.gate_proj.weight"]
        init_info.dtype_names = ["bfloat16", "bfloat16"]
        init_info.shapes = [[4, 8], [16, 8]]
        payload = init_info.to_api_payload()
        assert set(payload) == {
            "trainer_actor_name",
            "trainer_actor_namespace",
            "produce_method_name",
            "names",
            "dtype_names",
            "shapes",
            "warmup_method_name",
        }
        # override_existing_receiver is SkyRL-only; must NOT leak into the engine payload.
        assert "override_existing_receiver" not in payload
        assert payload["names"] == init_info.names
        assert payload["produce_method_name"] == "rdt_produce_weights_batched"

    def test_create_receiver_not_supported(self):
        """RDT has no SkyRL-side receiver (the receiver is the vLLM engine)."""
        init_info = ShardedRdtTransferStrategy.create_init_info(self._make_ie_cfg())
        with pytest.raises(NotImplementedError):
            ShardedRdtTransferStrategy.create_receiver(init_info)

    def test_populate_init_info_fills_metadata(self):
        """populate_init_info pulls names/dtypes/shapes from the trainer's extractor."""
        init_info = ShardedRdtTransferStrategy.create_init_info(self._make_ie_cfg())
        assert init_info.names == []  # empty until populated

        class _FakeExtractor:
            def get_weight_metadata(self, dtype):
                return {
                    "names": ["model.embed_tokens.weight", "lm_head.weight"],
                    "dtype_names": ["bfloat16", "bfloat16"],
                    "shapes": [[4, 8], [8, 4]],
                }

        ShardedRdtTransferStrategy.populate_init_info(init_info, weight_extractor=_FakeExtractor())
        assert init_info.names == ["model.embed_tokens.weight", "lm_head.weight"]
        assert init_info.dtype_names == ["bfloat16", "bfloat16"]
        assert init_info.shapes == [[4, 8], [8, 4]]

    def test_initialize_receivers_routes_to_rdt_endpoint(self):
        """initialize_receivers calls the RDT collective_rpc, not the NCCL init."""
        init_info = ShardedRdtTransferStrategy.create_init_info(self._make_ie_cfg())
        init_info.names = ["a"]
        init_info.dtype_names = ["bfloat16"]
        init_info.shapes = [[1]]
        sentinel = object()
        captured = {}

        class _FakeClient:
            def init_weight_transfer_engine_rdt(self, payload):
                captured["payload"] = payload
                return sentinel

            def init_weight_update_communicator(self, info):
                raise AssertionError("RDT must not use the native NCCL init path")

        ret = ShardedRdtTransferStrategy.initialize_receivers(init_info, _FakeClient())
        assert ret is sentinel
        assert captured["payload"]["names"] == ["a"]
        assert "override_existing_receiver" not in captured["payload"]


class TestShardedRdtVllmRegistration:
    """The monkeypatch + factory registration (requires the vLLM wheel)."""

    def test_engine_registered_and_config_relaxed(self):
        pytest.importorskip("vllm")
        # Importing the weight_sync package runs rdt_vllm_register.ensure_registered().
        from vllm.config import WeightTransferConfig
        from vllm.distributed.weight_transfer import WeightTransferEngineFactory

        import skyrl.backends.skyrl_train.weight_sync  # noqa: F401

        assert "sharded_rdt" in WeightTransferEngineFactory._registry
        # Literal relaxed so the custom backend validates, without breaking the originals.
        assert WeightTransferConfig(backend="sharded_rdt").backend == "sharded_rdt"
        assert WeightTransferConfig(backend="nccl").backend == "nccl"
        assert WeightTransferConfig(backend="ipc").backend == "ipc"


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
        init_info = BroadcastTransferStrategy.create_init_info(ie_cfg, inference_world_size=4)

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
