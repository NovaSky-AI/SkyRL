import pytest

from skyrl.backends.skyrl_train.weight_sync import (
    BroadcastInitInfo,
    BroadcastTransferStrategy,
    BroadcastWeightUpdateRequest,
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    CudaIpcWeightUpdateRequest,
    LoraLoadRequest,
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
        ],
    )
    def test_returns_correct_strategy(self, backend, colocate_all, expected_strategy):
        """Should return correct strategy based on backend and colocate_all."""
        assert get_transfer_strategy_cls(backend, colocate_all) is expected_strategy

    def test_sharded_rdt_bypasses_strategy_layer(self):
        """sharded_rdt has no WeightTransferStrategy — it's driven directly by
        RdtWeightSyncSender from init_weight_sync_state, so the class selector must
        refuse it loudly rather than silently falling back to a push strategy."""
        with pytest.raises(ValueError):
            get_transfer_strategy_cls("sharded_rdt", False)

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


class TestRdtSend:
    """Tests for the sharded_rdt (NIXL pull) trainer-send glue — no GPU/vLLM.

    RDT bypasses the WeightTransferStrategy/Sender abstraction; the glue lives in
    ``weight_sync/rdt_send.py`` (``_FsdpWeightSource`` + ``_SyncInferenceClient``,
    driven by ``RdtWeightSyncSender``). These cover the group-major WeightSource
    reorder and the sync-client -> RDT-route mapping without touching the vendored
    engine (whose ``trainer_init`` needs Ray + GPU)."""

    def test_weight_source_reorders_group_major(self):
        """The FSDP WeightSource reorders metadata into group-major order (pre /
        per-layer / post) so the vendored trainer's group-contiguity check passes."""
        import torch

        from skyrl.backends.skyrl_train.weight_sync.rdt_send import _FsdpWeightSource

        class _FakeExtractor:
            weight_prefix = ""

            def get_weight_metadata(self, dtype):
                # Layer 1 before layer 0 => must reorder to pre / layer-0 / layer-1 / post.
                return {
                    "names": [
                        "model.embed_tokens.weight",
                        "model.layers.1.mlp.gate_proj.weight",
                        "model.layers.0.mlp.gate_proj.weight",
                        "lm_head.weight",
                    ],
                    "dtype_names": ["bfloat16", "bfloat16", "bfloat16", "bfloat16"],
                    "shapes": [[4, 8], [1, 8], [0, 8], [8, 4]],
                }

        source = _FsdpWeightSource(_FakeExtractor(), torch.bfloat16)
        meta = source.metadata()
        assert [m.name for m in meta] == [
            "model.embed_tokens.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.1.mlp.gate_proj.weight",
            "lm_head.weight",
        ]
        # shapes travel with their names through the reorder; dtype is the wire dtype.
        assert [list(m.shape) for m in meta] == [[4, 8], [0, 8], [1, 8], [8, 4]]
        assert all(m.dtype is torch.bfloat16 for m in meta)

    def test_sync_client_maps_to_rdt_routes(self):
        """The VLLMWeightSyncClient adapter routes the engine's four synchronous
        calls onto SkyRL's async RDT control-plane coroutines (never the native
        NCCL init) and runs them on the bound event loop."""
        import asyncio

        from skyrl.backends.skyrl_train.weight_sync.rdt_send import _SyncInferenceClient

        calls = []

        class _FakeClient:
            async def init_weight_transfer_engine_rdt(self, payload):
                calls.append(("init", payload))

            async def start_weight_update(self, is_checkpoint_format=False):
                calls.append(("start", is_checkpoint_format))

            async def update_weights_rdt(self, update_info):
                calls.append(("update", update_info))

            async def finish_weight_update(self):
                calls.append(("finish",))

            def init_weight_update_communicator(self, info):
                raise AssertionError("RDT must not use the native NCCL init path")

        async def _drive():
            loop = asyncio.get_running_loop()
            sc = _SyncInferenceClient(_FakeClient(), loop)
            # _run uses run_coroutine_threadsafe, so drive from a worker thread.
            await asyncio.to_thread(sc.init_weight_transfer_engine, {"names": ["w"]})
            await asyncio.to_thread(sc.start_weight_update)
            await asyncio.to_thread(sc.update_weights, {})
            await asyncio.to_thread(sc.finish_weight_update)

        asyncio.run(_drive())
        assert [c[0] for c in calls] == ["init", "start", "update", "finish"]
        assert calls[0][1] == {"names": ["w"]}
        assert calls[1][1] is True  # start_weight_update(is_checkpoint_format=True)


class TestRdtReplicaConsumerMapping:
    """The per-replica consumer identity the engine computes from the injected
    replica_rank/num_replicas must give every worker in a multi-engine fleet a
    DISTINCT global id and a correct 1:1 producer binding (the fix for the
    multi-engine deadlock). This mirrors the engine's arithmetic over the shared
    M:N helpers, so it runs without a GPU/vLLM."""

    @staticmethod
    def _consumer_id(replica_rank, num_replicas, num_consumers, local_index):
        # Mirrors ShardedRDTWeightTransferEngine.init_transfer_engine.
        workers_per_replica = num_consumers // max(1, num_replicas)
        return replica_rank * workers_per_replica + local_index

    def test_two_dense_engines_bind_distinct_producers(self):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import (
            assign_producer_indices,
            count_consumers,
        )

        # 2 independent TP=1 engines (the 2x2 e2e): each engine's local index is 0,
        # replica_rank 0 and 1 => consumer ids 0 and 1 (previously both 0 -> deadlock).
        num_consumers, num_producers, num_replicas = 2, 2, 2
        cids = [self._consumer_id(r, num_replicas, num_consumers, 0) for r in range(2)]
        assert cids == [0, 1]
        # Each consumer binds its own producer; each producer serves exactly one.
        assert assign_producer_indices(num_producers, num_consumers, cids[0]) == [0]
        assert assign_producer_indices(num_producers, num_consumers, cids[1]) == [1]
        assert count_consumers(num_producers, num_consumers, 0) == 1
        assert count_consumers(num_producers, num_consumers, 1) == 1

    def test_single_replica_offset_is_zero(self):
        # num_replicas=1 (default / single deployment) => offset 0, id == local index.
        assert self._consumer_id(0, 1, 4, 3) == 3

    def test_multi_engine_multi_worker_ids_are_contiguous(self):
        # 2 engines x TP=2 = 4 consumers; ids must cover 0..3 with no collision.
        num_consumers, num_replicas = 4, 2
        ids = [self._consumer_id(r, num_replicas, num_consumers, local) for r in range(2) for local in range(2)]
        assert sorted(ids) == [0, 1, 2, 3]


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
