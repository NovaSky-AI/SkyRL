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
    ``weight_sync/rdt_send.py`` (``_FsdpWeightSource`` driven by
    ``RdtWeightSyncSender``). This covers the group-major WeightSource reorder
    without touching the vendored engine (whose ``trainer_init`` needs Ray + GPU).
    The synchronous control-plane client is covered in ``test_rdt_control_plane``."""

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

    def test_megatron_weight_source_streams_bridge_export(self):
        """MegatronWeightSource wraps the extractor's Megatron-Bridge: metadata()
        and iteration both run the non-bucketed export (so their order agrees), and
        iteration casts each full tensor to the wire dtype."""
        import torch

        from skyrl.backends.skyrl_train.weight_sync.rdt_send import MegatronWeightSource
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import (
            layerwise_groups,
        )

        # HF-canonical (group-contiguous) order, fp32 source tensors.
        items = [
            ("model.embed_tokens.weight", torch.ones(4, 8, dtype=torch.float32)),
            ("model.layers.0.mlp.gate_proj.weight", torch.ones(2, 8, dtype=torch.float32)),
            ("model.layers.1.mlp.gate_proj.weight", torch.ones(2, 8, dtype=torch.float32)),
            ("lm_head.weight", torch.ones(8, 4, dtype=torch.float32)),
        ]
        export_calls = []

        class _FakeBridge:
            def export_hf_weights(self, module, show_progress=False, conversion_tasks=None):
                # RDT must use the NON-bucketed export (conversion_tasks=None) so
                # MoE-expert grouping doesn't break group-major contiguity.
                assert conversion_tasks is None
                export_calls.append(module)
                for name, tensor in items:
                    yield name, tensor

        class _FakeMegatronExtractor:
            bridge = _FakeBridge()
            actor_module = object()

        source = MegatronWeightSource(_FakeMegatronExtractor(), torch.bfloat16)

        meta = source.metadata()
        names = [m.name for m in meta]
        assert names == [n for n, _ in items]  # order preserved (no reorder)
        assert [list(m.shape) for m in meta] == [[4, 8], [2, 8], [2, 8], [8, 4]]
        assert all(m.dtype is torch.bfloat16 for m in meta)
        # Order is already group-contiguous -> layerwise_groups partitions it exactly
        # (this is what the trainer engine's trainer_init validates).
        assert [n for g in layerwise_groups(names) for n in g] == names

        yielded = list(source)
        assert [n for n, _ in yielded] == names
        assert all(t.dtype is torch.bfloat16 and t.is_contiguous() for _, t in yielded)
        # metadata() cached: iteration ran a fresh export, so the bridge was
        # exported twice total (one dry-run for metadata, one for the stream).
        assert len(export_calls) == 2

    def test_make_weight_source_selects_by_extractor_flavor(self):
        """make_weight_source picks Megatron (has .bridge) vs FSDP (has .model)."""
        import torch

        from skyrl.backends.skyrl_train.weight_sync.rdt_send import (
            MegatronWeightSource,
            _FsdpWeightSource,
            make_weight_source,
        )

        class _FakeBridge:
            def export_hf_weights(self, module, show_progress=False, conversion_tasks=None):
                return iter(())

        class _FakeMegatronExtractor:
            bridge = _FakeBridge()
            actor_module = object()

        class _FakeFsdpExtractor:
            weight_prefix = ""
            model = object()

            def get_weight_metadata(self, dtype):
                return {"names": [], "dtype_names": [], "shapes": []}

        assert isinstance(make_weight_source(_FakeMegatronExtractor(), torch.bfloat16), MegatronWeightSource)
        assert isinstance(make_weight_source(_FakeFsdpExtractor(), torch.bfloat16), _FsdpWeightSource)


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
            RdtRouter,
            assign_producer_indices,
        )

        # 2 independent TP=1 engines (the 2x2 e2e): each engine's local index is 0,
        # replica_rank 0 and 1 => consumer ids 0 and 1 (previously both 0 -> deadlock).
        num_consumers, num_producers, num_replicas = 2, 2, 2
        cids = [self._consumer_id(r, num_replicas, num_consumers, 0) for r in range(2)]
        assert cids == [0, 1]
        # Each consumer binds its own producer; each producer serves exactly one.
        assert assign_producer_indices(num_producers, num_consumers, cids[0]) == [0]
        assert assign_producer_indices(num_producers, num_consumers, cids[1]) == [1]
        router = RdtRouter(num_producers, num_consumers, None, num_groups=3)
        assert router.free_target(0, 0) == 1
        assert router.free_target(1, 0) == 1

    def test_single_replica_offset_is_zero(self):
        # num_replicas=1 (default / single deployment) => offset 0, id == local index.
        assert self._consumer_id(0, 1, 4, 3) == 3

    def test_multi_engine_multi_worker_ids_are_contiguous(self):
        # 2 engines x TP=2 = 4 consumers; ids must cover 0..3 with no collision.
        num_consumers, num_replicas = 4, 2
        ids = [self._consumer_id(r, num_replicas, num_consumers, local) for r in range(2) for local in range(2)]
        assert sorted(ids) == [0, 1, 2, 3]


class TestRdtRouter:
    """Who serves and frees each gather group.

    A wrong answer here is not a wrong number but a hang: a consumer pulling from
    a producer that never gathered a group waits forever, and a published group
    nobody frees stalls the producer's end_sync. So every case checks the
    conservation law that makes the credit loop terminate — for each group, the
    per-producer free targets sum to the consumer count."""

    @staticmethod
    def _conserved(router, num_groups):
        return all(
            sum(router.free_target(p, g) for p in router.owners(g)) == router.num_consumers for g in range(num_groups)
        )

    def test_identity_when_fleets_match(self):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import RdtRouter

        r = RdtRouter(8, 8, None, num_groups=6)
        assert [r.bound_producers(c) for c in range(8)] == [[c] for c in range(8)]
        assert all(r.producer_for(3, g) == 3 for g in range(6))
        assert self._conserved(r, 6)

    def test_gather_to_all_keeps_the_historical_binding(self):
        """16 producers / 8 consumers: the same producers per consumer as the
        pre-router block rule, but each group is pulled from ONE of them."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import (
            RdtRouter,
            assign_producer_indices,
        )

        r = RdtRouter(16, 8, None, num_groups=95)
        for c in range(8):
            assert r.bound_producers(c) == assign_producer_indices(16, 8, c)
        assert [r.producer_for(0, g) for g in range(4)] == [0, 1, 0, 1]
        # No producer is left publishing groups nobody pulls from it.
        assert {r.producer_for(c, g) for c in range(8) for g in range(95)} == set(range(16))
        assert self._conserved(r, 95)

    def test_fan_in_shares_one_producer(self):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import RdtRouter

        r = RdtRouter(2, 8, None, num_groups=5)
        assert [r.bound_producers(c) for c in range(8)] == [[c // 4] for c in range(8)]
        assert r.free_target(0, 0) == 4 and r.free_target(1, 0) == 4
        assert self._conserved(r, 5)

    def test_pipeline_stages_route_to_the_owning_stage(self):
        """2 stages x 8 ranks (the 235B tp4/pp2/ep8 -> TP8 shape): groups 0-2 on
        stage 0, 3-5 on stage 1. Each consumer must reach both stages, pulling
        each group from its owner."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import RdtRouter

        owners = [list(range(8))] * 3 + [list(range(8, 16))] * 3
        r = RdtRouter(16, 8, owners)
        for c in range(8):
            assert r.bound_producers(c) == [c, c + 8]
            assert [r.producer_for(c, g) for g in range(6)] == [c] * 3 + [c + 8] * 3
        assert r.owned_groups(0) == [0, 1, 2]
        assert r.owned_groups(8) == [3, 4, 5]
        # A rank owning a group serves exactly one consumer; non-owners serve none.
        assert r.free_target(0, 0) == 1 and r.free_target(0, 3) == 0
        assert self._conserved(r, 6)
        r.validate()

    def test_owner_without_a_consumer_gets_a_zero_target(self):
        """Fewer consumers than a stage has ranks: some owners serve nothing and
        must not publish (the trainer skips those groups)."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import RdtRouter

        r = RdtRouter(8, 2, [list(range(8))] * 4)
        r.validate()
        assert self._conserved(r, 4)
        assert any(
            r.free_target(p, g) == 0 for p in range(8) for g in range(4)
        ), "expected some (producer, group) pairs to serve no consumer"

    def test_validate_rejects_an_unowned_group(self):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import RdtRouter

        with pytest.raises(ValueError, match="no owner"):
            RdtRouter(4, 2, [[0, 1], [], [2, 3]]).validate()

    def test_validate_rejects_an_out_of_range_owner(self):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import RdtRouter

        with pytest.raises(ValueError, match="out of range"):
            RdtRouter(2, 2, [[0, 5]]).validate()


class TestRdtRouterLiveConsumers:
    """``free_target(..., live_consumer_ids=...)`` — the whole producer-side
    mechanism for syncing to a fleet that has lost an inference engine.

    The property that makes degradation safe is that ``producer_for`` is pure in
    the consumer id: dropping consumers cannot change any SURVIVING consumer's
    binding, so a live target is always a subset-count of the provisioned one and
    a producer never has to serve a name it did not register at init. These pin
    that, across the routing shapes the real deployments use."""

    ROUTERS = {
        # (num_producers, num_consumers, group_owners, num_groups)
        "identity_8x8": (8, 8, None, 6),
        "gather_to_all_16x8": (16, 8, None, 95),
        "fan_in_2x8": (2, 8, None, 5),
        "pp_local_2stage": (16, 8, [list(range(8))] * 3 + [list(range(8, 16))] * 3, 6),
        "owners_exceed_consumers": (8, 2, [list(range(8))] * 4, 4),
    }

    @staticmethod
    def _router(spec):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import RdtRouter

        p, c, owners, ngroups = spec
        return RdtRouter(p, c, owners, num_groups=ngroups), ngroups

    @pytest.mark.parametrize("name", list(ROUTERS))
    def test_full_live_set_is_identical_to_no_live_set(self, name):
        """Passing every consumer must be byte-identical to passing None, so an
        FT-enabled run that never loses an engine behaves exactly like today."""
        r, ngroups = self._router(self.ROUTERS[name])
        everyone = list(range(r.num_consumers))
        for g in range(ngroups):
            for p in r.owners(g):
                assert r.free_target(p, g, everyone) == r.free_target(p, g)

    @pytest.mark.parametrize("name", list(ROUTERS))
    def test_live_targets_match_a_direct_count_over_producer_for(self, name):
        """Cross-check against the definition, over hole-y live sets — the live set
        is not required to be a prefix or contiguous (a restarted-slot fleet in
        Part 2 will not be)."""
        r, ngroups = self._router(self.ROUTERS[name])
        c = r.num_consumers
        for live in ({0}, set(range(c)) - {0}, set(range(0, c, 2)), {c - 1}, set(range(c))):
            for g in range(ngroups):
                for p in r.owners(g):
                    expected = sum(1 for x in sorted(live) if r.producer_for(x, g) == p)
                    assert r.free_target(p, g, live) == expected

    @pytest.mark.parametrize("name", list(ROUTERS))
    def test_live_targets_never_exceed_provisioned(self, name):
        r, ngroups = self._router(self.ROUTERS[name])
        live = set(range(r.num_consumers)) - {0}
        for g in range(ngroups):
            for p in r.owners(g):
                assert r.free_target(p, g, live) <= r.free_target(p, g)

    @pytest.mark.parametrize("name", list(ROUTERS))
    def test_live_targets_still_conserve_over_the_live_set(self, name):
        """The termination law, restated for a degraded sync: every live consumer
        frees each group exactly once, so the per-producer targets sum to the LIVE
        count. Anything else is a hang (short) or a double-free (long)."""
        r, ngroups = self._router(self.ROUTERS[name])
        live = sorted(set(range(r.num_consumers)) - {0})
        for g in range(ngroups):
            assert sum(r.free_target(p, g, live) for p in r.owners(g)) == len(live)

    def test_surviving_consumers_keep_their_bindings(self):
        """The property degradation rests on: removing a consumer must not move any
        other consumer's producer for any group."""
        r, ngroups = self._router(self.ROUTERS["gather_to_all_16x8"])
        before = {(c, g): r.producer_for(c, g) for c in range(r.num_consumers) for g in range(ngroups)}
        live = sorted(set(range(r.num_consumers)) - {3})
        after = {(c, g): r.producer_for(c, g) for c in live for g in range(ngroups)}
        assert all(after[k] == before[k] for k in after)

    def test_an_entirely_dead_consumer_block_zeroes_its_producers(self):
        """2 producers, 8 consumers: consumers 0-3 pull from producer 0. Kill all
        four and producer 0 has nothing to publish — it still runs every gather
        collective, but its groups become gather-and-drop."""
        r, ngroups = self._router(self.ROUTERS["fan_in_2x8"])
        live = [4, 5, 6, 7]
        for g in range(ngroups):
            assert r.free_target(0, g, live) == 0
            assert r.free_target(1, g, live) == 4

    def test_empty_live_set_zeroes_everything(self):
        r, ngroups = self._router(self.ROUTERS["identity_8x8"])
        assert all(r.free_target(p, g, []) == 0 for g in range(ngroups) for p in r.owners(g))

    def test_validate_still_runs_over_the_provisioned_set(self):
        """``validate()`` asserts a PROVISIONING invariant (targets sum to
        num_consumers), not a per-sync one, so it must keep ignoring liveness."""
        r, _ = self._router(self.ROUTERS["pp_local_2stage"])
        r.validate()  # unchanged by anything above


class TestPpLocalOwnership:
    """``MegatronStackedWeightSource`` ownership detection (SKYRL_RDT_PP_LOCAL).

    In PP-local mode a stage exports only its own parameters, so the source has
    to (a) rebuild WHOLE-model metadata from what the stages exchange — the RDT
    contract requires every rank to describe the whole model — and (b) notice when
    one gather group is produced by two stages, which cannot be served per-stage.
    Both are exercised against the assembly directly (the walk itself needs
    Megatron + GPUs)."""

    @staticmethod
    def _source(pp_size, my_pp, gathered):
        """A source in PP-local mode with the stages' exchange stubbed out.

        ``metadata()`` is pre-populated the way the real one does it (the walk's
        local result handed to ``_assemble_pp_metadata``), so the assembly and
        ``owned_groups`` can be exercised without Megatron or a GPU."""
        import torch

        from skyrl.backends.skyrl_train.weight_sync.rdt_send import (
            MegatronStackedWeightSource,
        )

        src = MegatronStackedWeightSource.__new__(MegatronStackedWeightSource)
        src._dtype = torch.bfloat16
        src._pp_local = True
        src._geo_cache = src._geo_sig = None
        src._group_stages = []
        src._owned_group_idx = []
        src._pp_geometry = lambda: (pp_size, my_pp)  # type: ignore[method-assign]
        src._exchange_pp_names = lambda mine: gathered  # type: ignore[method-assign]
        src._meta = src._assemble_pp_metadata([])
        return src

    def test_metadata_is_the_whole_model_group_major_on_every_stage(self):
        """Stage 1 walks only its own layer, but metadata() must come back as the
        whole model in group-major order — identical on both stages, since the
        engine cross-checks a digest of it and bakes the consumers' plan from one
        rank's copy."""
        stage0 = [("model.embed_tokens.weight", [8, 4]), ("model.layers.0.w", [4, 4])]
        stage1 = [("model.layers.1.w", [4, 4]), ("model.norm.weight", [4])]
        expected = [
            "model.embed_tokens.weight",
            "model.layers.0.w",
            "model.layers.1.w",
            "model.norm.weight",
        ]
        for my_pp in (0, 1):
            meta = self._source(2, my_pp, [stage0, stage1]).metadata()
            assert [m.name for m in meta] == expected
            assert [tuple(m.shape) for m in meta] == [(8, 4), (4, 4), (4, 4), (4,)]
        # ... and each stage claims exactly the groups it produced.
        assert self._source(2, 0, [stage0, stage1]).owned_groups() == [0, 1]
        assert self._source(2, 1, [stage0, stage1]).owned_groups() == [2, 3]

    def test_metadata_is_group_contiguous_even_when_a_stage_holds_both_ends(self):
        """The assembled order must satisfy the engine's group-contiguity check —
        ``flat(layerwise_groups(names)) == names`` — for whatever the stages
        produce, and must be the same list on every stage.

        Here stage 0 holds the output block too (a tied-embedding layout), so it
        yields two non-layer names before any layer exists. ``layerwise_groups``
        splits pre/post by POSITION, so those land in one leading group rather
        than a pre and a post block. That is still a valid partition — ownership
        follows it, and both sides derive it from the same list — which is why the
        invariant to hold is contiguity, not a canonical pre/layers/post shape."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_common import (
            layerwise_groups,
        )

        stage0 = [("model.embed_tokens.weight", [8, 4]), ("model.lm_head.weight", [8, 4])]
        stage1 = [("model.layers.0.w", [4, 4])]
        per_stage = []
        for my_pp in (0, 1):
            src = self._source(2, my_pp, [stage0, stage1])
            names = [m.name for m in src.metadata()]
            assert [n for g in layerwise_groups(names) for n in g] == names
            per_stage.append(names)
        assert per_stage[0] == per_stage[1]
        # Stage 0 produced both names of the leading group; stage 1 the layer.
        src = self._source(2, 1, [stage0, stage1])
        assert src.owned_groups() == [1]
        assert src._group_stages == [{0}, {1}]

    def test_a_group_produced_by_two_stages_disables_pp_local(self):
        """Tied embeddings / MTP put one group's names on two stages. Serving that
        per-stage would publish half a group, so the source must fall back to
        gather-to-all instead of silently truncating it."""
        stage0 = [("model.embed_tokens.weight", [8, 4]), ("model.layers.0.w", [4, 4])]
        # Stage 1 also produces a post-block name -> the post group spans stages.
        stage1 = [("model.layers.1.w", [4, 4]), ("model.norm.weight", [4])]
        stage0 = stage0 + [("model.lm_head.weight", [8, 4])]
        src = self._source(2, 0, [stage0, stage1])
        assert src.owned_groups() is None
        assert src._pp_local is False
        # The stale local-only expert geometry must not survive the fallback: a
        # gather-to-all walk needs every layer, and its cache key would not change.
        assert src._geo_cache is None and src._geo_sig is None
        # Metadata is still the whole model, so the digest check still passes.
        assert [m.name for m in src.metadata()] == [
            "model.embed_tokens.weight",
            "model.layers.0.w",
            "model.layers.1.w",
            "model.lm_head.weight",
            "model.norm.weight",
        ]

    def test_a_tied_name_on_two_stages_is_not_duplicated(self):
        """A weight both stages hold (Megatron keeps a copy of a tied embedding on
        the last stage) must appear ONCE in metadata — a duplicate name would give
        the consumers two plan entries for one tensor — and mark its group shared."""
        tied = ("model.embed_tokens.weight", [8, 4])
        src = self._source(2, 0, [[tied, ("model.layers.0.w", [4, 4])], [tied]])
        assert [m.name for m in src.metadata()] == ["model.embed_tokens.weight", "model.layers.0.w"]
        assert src._group_stages[0] == {0, 1}
        assert src.owned_groups() is None

    def test_walk_is_reordered_into_partition_order(self):
        """The bridge streams a stage's tasks in ITS order, which need not match the
        partition: at 235B the last stage exports the output block BEFORE its layers,
        while layerwise_groups places that block last. The gather loop walks
        owned_groups() ascending and raises on anything else, so the walk has to be
        reordered — this is the bug the 235B bench caught (48+48 groups against a
        95-group reference) before it could fail mid-sync."""
        stage0 = [("model.embed_tokens.weight", [8, 4]), ("model.layers.0.w", [4, 4])]
        stage1 = [("model.norm.weight", [4]), ("model.layers.1.w", [4, 4])]
        src = self._source(2, 1, [stage0, stage1])
        assert src.owned_groups() == [2, 3]  # layer 1, then post

        # Stage 1's walk emits the post block first, as the real bridge does.
        walk = iter([(["model.norm.weight"], ["N"]), (["model.layers.1.w"], ["L1"])])
        assert list(src._walk_in_group_order(walk)) == [
            (["model.layers.1.w"], ["L1"]),
            (["model.norm.weight"], ["N"]),
        ]

    def test_walk_reorder_refuses_to_hold_layer_stacks(self):
        """Deferring a group pins its gathered tensors (~4.6 GiB for a 235B layer),
        so an unexpected permutation must raise rather than quietly inflate trainer
        memory."""
        gathered = [[(f"model.layers.{i}.w", [4, 4]) for i in range(5)], []]
        src = self._source(2, 0, gathered)
        assert src.owned_groups() == [0, 1, 2, 3, 4]
        # Every group arrives in reverse: nothing can be released.
        walk = iter([([f"model.layers.{i}.w"], [i]) for i in (4, 3, 2, 1, 0)])
        with pytest.raises(RuntimeError, match="groups ahead of the partition order"):
            list(src._walk_in_group_order(walk))

    def test_walk_reorder_rejects_a_name_outside_the_partition(self):
        stage0 = [("model.layers.0.w", [4, 4])]
        src = self._source(1, 0, [stage0])
        walk = iter([(["model.layers.9.w"], ["X"])])
        with pytest.raises(RuntimeError, match="not in the assembled partition"):
            list(src._walk_in_group_order(walk))

    def test_pp_local_is_on_by_default_and_opt_out_only(self, monkeypatch):
        """PP-local is the default; only an explicit ``0`` forces gather-to-all.

        It engages only at pp > 1 and is self-healing, so the opt-out exists as an
        escape hatch rather than a knob anyone is expected to set."""
        from skyrl.backends.skyrl_train.weight_sync import rdt_send

        monkeypatch.delenv("SKYRL_RDT_PP_LOCAL", raising=False)
        assert rdt_send._pp_local_requested() is True
        monkeypatch.setenv("SKYRL_RDT_PP_LOCAL", "1")
        assert rdt_send._pp_local_requested() is True
        monkeypatch.setenv("SKYRL_RDT_PP_LOCAL", "0")
        assert rdt_send._pp_local_requested() is False


class TestQkvIndexDeviceCtx:
    """``_qkv_index_device_ctx`` keeps the QKV split's index tensors on the weight's
    device instead of the host, which is worth ~0.65s/sync of ne_bridge at 235B (a
    CPU index tensor against a CUDA weight forces an H2D copy + stream sync per
    gather). It deliberately copies NO upstream logic — it only changes where
    ``torch.arange`` allocates — so these cover the wrapping, the injection, and
    the restore. A meta device stands in for CUDA so this needs no GPU."""

    @staticmethod
    def _fake_modules(monkeypatch):
        """Stub split fns on the real modules, so the context wraps something we can
        observe. Records the device each torch.arange call landed on."""
        import torch
        from megatron.bridge.models.conversion import param_mapping as pm

        seen = []

        def _split(config, qkv, *a, **kw):
            seen.append(torch.arange(4).device.type)
            return ("q", "k", "v")

        for name in ("split_qkv_weights", "split_qkv_biases", "split_qkv_weights_scale"):
            monkeypatch.setattr(pm, name, _split, raising=False)
        return pm, seen

    def test_index_tensors_follow_the_weight_device(self, monkeypatch):
        import torch

        pytest.importorskip("megatron.bridge.models.conversion.param_mapping")
        from skyrl.backends.skyrl_train.weight_sync import rdt_send

        pm, seen = self._fake_modules(monkeypatch)
        weight = torch.empty(2, 2, device="meta")
        with rdt_send._qkv_index_device_ctx():
            pm.split_qkv_weights(None, weight)
        assert seen == ["meta"], "arange should have been redirected to the weight's device"

    def test_cpu_weights_are_left_alone(self, monkeypatch):
        """The redirect must not fire for a host weight — there is nothing to fix and
        forcing a device would be a behaviour change."""
        import torch

        pytest.importorskip("megatron.bridge.models.conversion.param_mapping")
        from skyrl.backends.skyrl_train.weight_sync import rdt_send

        pm, seen = self._fake_modules(monkeypatch)
        with rdt_send._qkv_index_device_ctx():
            pm.split_qkv_weights(None, torch.empty(2, 2))
        assert seen == ["cpu"]

    def test_originals_and_torch_arange_are_restored(self, monkeypatch):
        """torch.arange is patched process-wide for the duration of ONE call, so a
        leak would silently put every later index tensor on a device."""
        import torch

        pytest.importorskip("megatron.bridge.models.conversion.param_mapping")
        from skyrl.backends.skyrl_train.weight_sync import rdt_send

        pm, _seen = self._fake_modules(monkeypatch)
        before, real_arange = pm.split_qkv_weights, torch.arange
        with rdt_send._qkv_index_device_ctx():
            assert pm.split_qkv_weights is not before  # wrapped
            pm.split_qkv_weights(None, torch.empty(2, 2, device="meta"))
            assert torch.arange is real_arange, "arange must be restored after each call"
        assert pm.split_qkv_weights is before
        assert torch.arange is real_arange
        assert torch.arange(3).device.type == "cpu"

    def test_disabled_by_env(self, monkeypatch):
        import torch

        pytest.importorskip("megatron.bridge.models.conversion.param_mapping")
        from skyrl.backends.skyrl_train.weight_sync import rdt_send

        monkeypatch.setenv("SKYRL_RDT_QKV_DEVICE_FIX", "0")
        pm, seen = self._fake_modules(monkeypatch)
        with rdt_send._qkv_index_device_ctx():
            pm.split_qkv_weights(None, torch.empty(2, 2, device="meta"))
        assert seen == ["cpu"], "with the fix off, arange must keep its default device"


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
