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

    def _make_ie_cfg(self, run_engines_locally: bool = False) -> InferenceEngineConfig:
        return InferenceEngineConfig(
            weight_sync_backend="sharded_rdt",
            model_dtype="bfloat16",
            run_engines_locally=run_engines_locally,
        )

    def test_create_init_info(self):
        """create_init_info returns a ShardedRdtInitInfo with the trainer actor name."""
        init_info = ShardedRdtTransferStrategy.create_init_info(self._make_ie_cfg(), inference_world_size=4)
        assert isinstance(init_info, ShardedRdtInitInfo)
        assert init_info.trainer_actor_name == RDT_TRAINER_ACTOR_NAME
        assert init_info.produce_method_name == "rdt_produce_weights_batched"
        assert init_info.num_consumers == 4
        # Non-local engines => override an existing receiver.
        assert init_info.override_existing_receiver is True
        # names/dtype_names/shapes/group_lens are filled later from the extractor.
        assert init_info.names == []
        assert init_info.group_lens == []
        assert init_info.strategy_type() is ShardedRdtTransferStrategy

    def test_create_init_info_requires_world_size(self):
        """The consumer count is mandatory (free ref-count target + M:N split)."""
        with pytest.raises(ValueError):
            ShardedRdtTransferStrategy.create_init_info(self._make_ie_cfg())

    def test_create_init_info_override_disabled_for_local_engines(self):
        init_info = ShardedRdtTransferStrategy.create_init_info(
            self._make_ie_cfg(run_engines_locally=True), inference_world_size=1
        )
        assert init_info.override_existing_receiver is False

    def test_to_api_payload(self):
        """to_api_payload matches the vLLM ShardedRDTWeightTransferInitInfo fields."""
        init_info = ShardedRdtTransferStrategy.create_init_info(self._make_ie_cfg(), inference_world_size=2)
        init_info.names = ["model.embed_tokens.weight", "model.layers.0.mlp.gate_proj.weight"]
        init_info.dtype_names = ["bfloat16", "bfloat16"]
        init_info.shapes = [[4, 8], [16, 8]]
        init_info.group_lens = [1, 1]
        payload = init_info.to_api_payload()
        assert set(payload) == {
            "trainer_actor_name",
            "trainer_actor_namespace",
            "produce_method_name",
            "num_consumers",
            "num_rdt_buffers",
            "layerwise_split",
            "arena_presize_gb",
            "pack_check",
            "names",
            "dtype_names",
            "shapes",
            "group_lens",
        }
        # SkyRL-only / producer-only fields must NOT leak into the engine payload.
        assert "override_existing_receiver" not in payload
        assert "gather_lookahead" not in payload
        assert payload["names"] == init_info.names
        assert payload["num_consumers"] == 2
        assert payload["produce_method_name"] == "rdt_produce_weights_batched"

    def test_populate_init_info_fills_group_major_metadata(self):
        """populate_init_info pulls names/dtypes/shapes from the extractor and
        reorders them into group-major order with a matching group_lens."""
        init_info = ShardedRdtTransferStrategy.create_init_info(self._make_ie_cfg(), inference_world_size=1)
        assert init_info.names == []  # empty until populated

        class _FakeExtractor:
            def get_weight_metadata(self, dtype):
                # Layer 1 before layer 0 => layerwise_groups must reorder them
                # (pre / layer-0 / layer-1 / post) and carry dtypes/shapes along.
                return {
                    "names": [
                        "model.embed_tokens.weight",
                        "model.layers.1.mlp.gate_proj.weight",
                        "model.layers.0.mlp.gate_proj.weight",
                        "lm_head.weight",
                    ],
                    "dtype_names": ["bfloat16", "float16", "bfloat16", "float16"],
                    "shapes": [[4, 8], [1, 8], [0, 8], [8, 4]],
                }

        ShardedRdtTransferStrategy.populate_init_info(init_info, weight_extractor=_FakeExtractor())
        # pre (embed) -> layer 0 -> layer 1 -> post (lm_head): group-major.
        assert init_info.names == [
            "model.embed_tokens.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
            "lm_head.weight",
        ]
        # dtypes/shapes travel with their names through the reorder.
        assert init_info.dtype_names == ["bfloat16", "bfloat16", "float16", "float16"]
        assert init_info.shapes == [[4, 8], [0, 8], [1, 8], [8, 4]]
        # 4 groups (pre / layer-0 / layer-1 / post), one name each.
        assert init_info.group_lens == [1, 1, 1, 1]
        assert sum(init_info.group_lens) == len(init_info.names)

    def test_initialize_receivers_routes_to_rdt_endpoint(self):
        """initialize_receivers calls the RDT collective_rpc, not the NCCL init."""
        init_info = ShardedRdtTransferStrategy.create_init_info(self._make_ie_cfg(), inference_world_size=1)
        init_info.names = ["a"]
        init_info.dtype_names = ["bfloat16"]
        init_info.shapes = [[1]]
        init_info.group_lens = [1]
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
    """The factory registration (requires the vLLM wheel)."""

    def test_engine_registered(self):
        pytest.importorskip("vllm")
        # Importing the weight_sync package's register module runs ensure_registered().
        from vllm.config import WeightTransferConfig
        from vllm.distributed.weight_transfer import WeightTransferEngineFactory

        from skyrl.backends.skyrl_train.weight_sync import rdt_vllm_register

        rdt_vllm_register.ensure_registered()
        assert "sharded_rdt" in WeightTransferEngineFactory._registry
        # vLLM 0.23.0 already accepts arbitrary backend strings (Literal | str);
        # no runtime relaxation needed, and the built-ins still validate.
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
