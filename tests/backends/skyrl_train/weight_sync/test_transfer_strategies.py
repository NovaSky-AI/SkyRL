import pytest

from skyrl.backends.skyrl_train.weight_sync import (
    BroadcastInitInfo,
    BroadcastTransferStrategy,
    BroadcastWeightUpdateRequest,
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    CudaIpcWeightUpdateRequest,
    DeltaInitInfo,
    DeltaTransferStrategy,
    LoraLoadRequest,
    get_transfer_strategy,
    get_transfer_strategy_cls,
)
from skyrl.train.config import InferenceEngineConfig
from skyrl.train.config.config import DeltaWeightSyncConfig


class TestGetTransferStrategyCls:
    """Tests for get_transfer_strategy_cls function."""

    @pytest.mark.parametrize(
        "backend,colocate_all,expected_strategy",
        [
            ("nccl", True, CudaIpcTransferStrategy),
            ("nccl", False, BroadcastTransferStrategy),
            ("gloo", True, BroadcastTransferStrategy),
            ("gloo", False, BroadcastTransferStrategy),
            ("delta", True, DeltaTransferStrategy),
            ("delta", False, DeltaTransferStrategy),
        ],
    )
    def test_returns_correct_strategy(self, backend, colocate_all, expected_strategy):
        """Should return correct strategy based on backend and colocate_all."""
        assert get_transfer_strategy_cls(backend, colocate_all) is expected_strategy

    def test_sharded_rdt_selects_its_own_strategy(self):
        """sharded_rdt goes through the strategy layer like every other backend —
        it must never silently fall back to a push strategy."""
        from skyrl.backends.skyrl_train.weight_sync import ShardedRdtTransferStrategy

        assert get_transfer_strategy_cls("sharded_rdt", False) is ShardedRdtTransferStrategy
        # colocate_all is rejected elsewhere (build_vllm_cli_args); selection must
        # still not hand it a push strategy.
        assert get_transfer_strategy_cls("sharded_rdt", True) is ShardedRdtTransferStrategy

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
    """The sharded_rdt trainer-send glue in ``weight_sync/sharded_rdt/rdt_send.py``
    (the ``WeightSource`` implementations driven by ``RdtWeightSyncSender``), covered
    without the vendored engine, whose ``trainer_init`` needs Ray + GPU."""

    def test_weight_source_reorders_group_major(self):
        """The FSDP WeightSource reorders metadata into group-major order (pre /
        per-layer / post) so the vendored trainer's group-contiguity check passes."""
        import torch

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            _FsdpWeightSource,
        )

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

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            MegatronWeightSource,
        )
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_base import (
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

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
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
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_common import (
            RdtRouter,
            assign_producer_indices,
        )

        # 2 independent TP=1 engines: each engine's local index is 0, so the
        # replica_rank offset is what separates them into consumer ids 0 and 1.
        num_consumers, num_producers, num_replicas = 2, 2, 2
        cids = [self._consumer_id(r, num_replicas, num_consumers, 0) for r in range(2)]
        assert cids == [0, 1]
        # Each consumer binds its own producer; each producer serves exactly one.
        assert assign_producer_indices(num_producers, num_consumers, cids[0]) == [0]
        assert assign_producer_indices(num_producers, num_consumers, cids[1]) == [1]
        names = ["g0.w", "g1.w", "g2.w"]
        router = RdtRouter(num_producers, num_consumers, None, None, names, [1, 1, 1])
        assert router.producer_for(cids[0], "g0.w") == 0
        assert router.producer_for(cids[1], "g0.w") == 1

    def test_single_replica_offset_is_zero(self):
        # num_replicas=1 (default / single deployment) => offset 0, id == local index.
        assert self._consumer_id(0, 1, 4, 3) == 3

    def test_multi_engine_multi_worker_ids_are_contiguous(self):
        # 2 engines x TP=2 = 4 consumers; ids must cover 0..3 with no collision.
        num_consumers, num_replicas = 4, 2
        ids = [self._consumer_id(r, num_replicas, num_consumers, local) for r in range(2) for local in range(2)]
        assert sorted(ids) == [0, 1, 2, 3]


class TestPpLocalOwnership:
    """``MegatronStackedWeightSource`` ownership detection (the PP grain of
    SKYRL_RDT_SHARD_AWARE).

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
        ``held_names`` can be exercised without Megatron or a GPU."""
        import torch

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            MegatronStackedWeightSource,
        )

        src = MegatronStackedWeightSource.__new__(MegatronStackedWeightSource)
        src._dtype = torch.bfloat16
        src._pp_local = True
        src._ep_local = True  # demotion must flip BOTH shard-aware grains
        src._ep_size = 1
        src._my_ep_rank = 0
        src._demoted = False
        src._verified = True
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
        # ... and each stage claims exactly the names it produced.
        assert self._source(2, 0, [stage0, stage1]).held_names() == [
            "model.embed_tokens.weight",
            "model.layers.0.w",
        ]
        assert self._source(2, 1, [stage0, stage1]).held_names() == [
            "model.layers.1.w",
            "model.norm.weight",
        ]

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
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_base import (
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
        assert src.held_names() == ["model.layers.0.w"]
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
        assert src.held_names() is None
        assert src._demoted is True
        assert src._pp_local is False
        # BOTH shard-aware grains demote together: a stamped name a rank no
        # longer serves per-stage would misroute pulls.
        assert src._ep_local is False
        # Re-asking is still None: a demoted source holds everything.
        assert src.held_names() is None
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
        assert src.held_names() is None

    def test_walk_is_reordered_into_partition_order(self):
        """The bridge streams a stage's tasks in ITS order, which need not match the
        partition: at 235B the last stage exports the output block BEFORE its layers,
        while layerwise_groups places that block last. The gather loop walks the held
        groups ascending and raises on anything else, so the walk must be reordered."""
        stage0 = [("model.embed_tokens.weight", [8, 4]), ("model.layers.0.w", [4, 4])]
        stage1 = [("model.norm.weight", [4]), ("model.layers.1.w", [4, 4])]
        src = self._source(2, 1, [stage0, stage1])
        # layer 1 and the post block, in partition order
        assert src.held_names() == ["model.layers.1.w", "model.norm.weight"]

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
        assert src.held_names() == [f"model.layers.{i}.w" for i in range(5)]
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

    def test_a_demoted_source_refuses_to_iterate(self):
        """A demoted source cannot serve (its gather paths are gone); iterating
        it is a wiring bug — make_weight_source should have delegated to the
        plain MegatronWeightSource."""
        src = self._source(2, 0, [[("a.weight", [2])], [("a.weight", [2])]])
        assert src.held_names() is None  # tied name on both stages -> demoted
        assert src._demoted is True
        with pytest.raises(RuntimeError, match="demoted"):
            list(src.iter_groups())


class TestExpertOwnership:
    """The EP grain of shard-aware serving: `held_names()` + the
    walk's real-vs-None emission must agree (the stamps are what the consumers
    route by, the Nones are what the trainer drops before publishing)."""

    @staticmethod
    def _stub(ep_size=2, my_ep_rank=1, pp_local=False):
        import torch

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            MegatronStackedWeightSource,
        )

        src = MegatronStackedWeightSource.__new__(MegatronStackedWeightSource)
        src._dtype = torch.bfloat16
        src._pp_local = pp_local
        src._ep_local = True
        src._ep_size = ep_size
        src._my_ep_rank = my_ep_rank
        src._demoted = False
        # These tests are about the EP ownership grain, not about name resolution:
        # the source has no bridge, so seed the per-layer expert-name cache with the
        # Qwen3-MoE layout the assertions below use. (Production always resolves
        # these through the bridge's mapping registry and raises if it cannot —
        # there is no synthesized fallback to lean on here.)
        src._expert_names = {
            layer: [
                name
                for e in range(ep_size * 2)
                for name in (
                    f"model.layers.{layer}.mlp.experts.{e}.gate_proj.weight",
                    f"model.layers.{layer}.mlp.experts.{e}.up_proj.weight",
                    f"model.layers.{layer}.mlp.experts.{e}.down_proj.weight",
                )
            ]
            for layer in range(8)
        }
        src._expert_name_source = "test stub"
        src._layer_geom = {}
        src._phase = {}
        src._phase_prefix = ""
        return src

    @staticmethod
    def _meta_of(names):
        import torch

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_base import (
            ParamMeta,
        )

        return [ParamMeta(n, torch.bfloat16, (2, 2)) for n in names]

    def test_stamps_are_expert_index_over_n_local_and_minus_one_elsewhere(self):
        src = self._stub(ep_size=2, my_ep_rank=1)
        src._meta = self._meta_of(
            ["embed.weight"]
            + [f"model.layers.0.mlp.experts.{e}.gate_proj.weight" for e in range(4)]
            + ["model.layers.0.input_layernorm.weight", "lm_head.weight"]
        )
        owners = src._name_owner()
        assert src._my_ep_rank == 1
        assert owners == [-1, 0, 0, 1, 1, -1, -1]

    def test_an_expert_count_not_divisible_by_ep_size_raises(self):
        src = self._stub(ep_size=2)
        src._meta = self._meta_of([f"model.layers.0.mlp.experts.{e}.up_proj.weight" for e in range(3)])
        with pytest.raises(RuntimeError, match="not divisible"):
            src._name_owner()

    def test_ep_local_off_returns_none(self):
        src = self._stub()
        src._ep_local = False
        src._pp_local = False
        assert src.held_names() is None

    def test_the_walk_materializes_exactly_the_stamped_experts(self):
        """The truthfulness clause of the ABC contract: within an owned layer, a
        name stamped my_ep_rank yields a real view of the LOCAL stack; every
        other expert's entries are None. Zero collectives on this path."""
        import torch

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            _ExpertLayer,
        )

        class _Task:
            def __init__(self, t):
                self.param_weight = t

        F, H, n_local = 3, 2, 2
        src = self._stub(ep_size=2, my_ep_rank=1)
        fc1 = [_Task(torch.full((2 * F, H), float(10 + i))) for i in range(n_local)]
        fc2 = [_Task(torch.full((H, F), float(20 + i))) for i in range(n_local)]
        lay = _ExpertLayer(layer=0, fc1=fc1, fc2=fc2, owned=True, n_local=n_local, F=F, H=H, ep_size=2)

        names, tensors = [], []
        src._extend_layer_experts(lay, None, names, tensors)

        E = 4
        assert len(names) == len(tensors) == 3 * E
        by_name = dict(zip(names, tensors))
        # Foreign coordinate (0): experts 0..1 are None.
        for e in (0, 1):
            for proj in ("gate_proj", "up_proj", "down_proj"):
                assert by_name[f"model.layers.0.mlp.experts.{e}.{proj}.weight"] is None
        # Own coordinate (1): experts 2..3 are real views with today's shapes.
        for i, e in enumerate((2, 3)):
            gate = by_name[f"model.layers.0.mlp.experts.{e}.gate_proj.weight"]
            up = by_name[f"model.layers.0.mlp.experts.{e}.up_proj.weight"]
            down = by_name[f"model.layers.0.mlp.experts.{e}.down_proj.weight"]
            assert gate.shape == (F, H) and up.shape == (F, H) and down.shape == (H, F)
            assert torch.equal(gate, torch.full((F, H), float(10 + i), dtype=torch.bfloat16))
            assert torch.equal(down, torch.full((H, F), float(20 + i), dtype=torch.bfloat16))

    def test_foreign_expert_shapes_are_synthesized_from_local_geometry(self):
        """metadata() cannot read .shape off a None; the walk records each
        layer's (F, H) before the expert names are emitted."""
        import torch

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            _ExpertLayer,
        )

        class _Task:
            def __init__(self, t):
                self.param_weight = t

        F, H = 3, 2
        src = self._stub(ep_size=2, my_ep_rank=0)
        lay = _ExpertLayer(
            layer=7,
            fc1=[_Task(torch.zeros((2 * F, H)))],
            fc2=[_Task(torch.zeros((H, F)))],
            owned=True,
            n_local=1,
            F=F,
            H=H,
            ep_size=2,
        )
        src._extend_layer_experts(lay, None, [], [])
        assert src._foreign_expert_shape("model.layers.7.mlp.experts.1.gate_proj.weight") == (F, H)
        assert src._foreign_expert_shape("model.layers.7.mlp.experts.1.down_proj.weight") == (H, F)


class TestHeldNamesComposition:
    """``held_names`` is the misroute guard's source of truth: exactly the names
    this rank publishes — its stage's groups, narrowed to the replicated names
    plus its own coordinate's experts. The trainer copies it verbatim into the
    ``served_names`` it hands the sidecar."""

    @staticmethod
    def _source(ep_size, my_ep_rank, names, *, pp_local=False, owned=None):
        import torch

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            MegatronStackedWeightSource,
        )
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_base import (
            ParamMeta,
        )

        src = MegatronStackedWeightSource.__new__(MegatronStackedWeightSource)
        src._dtype = torch.bfloat16
        src._pp_local = pp_local
        src._ep_local = ep_size > 1
        src._ep_size = ep_size
        src._my_ep_rank = my_ep_rank
        src._demoted = False
        src._expert_names = {}
        src._layer_geom = {}
        src._phase = {}
        src._phase_prefix = ""
        src._meta = [ParamMeta(n, torch.bfloat16, (2, 2)) for n in names]
        if owned is not None:
            src._owned_group_idx = owned
            src._group_stages = []
        return src

    def test_ep_local_holds_replicated_names_plus_its_own_experts(self):
        names = [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.norm.weight",
        ]
        src = self._source(2, 1, names)
        assert src.held_names() == [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
            "model.norm.weight",
        ]

    def test_the_other_coordinate_holds_the_complement(self):
        names = [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.1.gate_proj.weight",
        ]
        held0 = self._source(2, 0, names).held_names()
        held1 = self._source(2, 1, names).held_names()
        assert held0 == [names[0]] and held1 == [names[1]]
        assert sorted(held0 + held1) == sorted(names)

    def test_neither_grain_holds_everything(self):
        names = ["a", "model.layers.0.w", "b"]
        assert self._source(1, 0, names).held_names() is None


@pytest.mark.vllm
class TestStampedYieldValidation:
    """The gather loop checks stamps against yields per group — the ABC's
    truthfulness invariant, enforced where both sit side by side. Without it a
    stamps/yield mismatch is a 300s stall-watchdog death instead of an
    immediate, named error."""

    @staticmethod
    def _engine(held):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_trainer import (
            ShardedRDTTrainerWeightTransferEngine,
        )

        e = ShardedRDTTrainerWeightTransferEngine.__new__(ShardedRDTTrainerWeightTransferEngine)
        e._held_names = held
        return e

    def test_matching_yields_pass(self):
        import torch

        e = self._engine({"norm", "e1"})
        e._validate_held_yields(0, ["norm", "e0", "e1"], [torch.zeros(1), None, torch.zeros(1)])

    def test_a_held_name_yielding_none_raises(self):
        """The dangerous direction: served_names advertises the name, consumers
        route pulls here, and the cache wait would never complete."""
        import torch

        e = self._engine({"norm", "e0"})
        with pytest.raises(RuntimeError, match="disagrees with the yielded tensors"):
            e._validate_held_yields(0, ["norm", "e0"], [torch.zeros(1), None])

    def test_a_foreign_name_yielding_a_tensor_raises(self):
        import torch

        e = self._engine({"norm"})
        with pytest.raises(RuntimeError, match="disagrees with the yielded tensors"):
            e._validate_held_yields(0, ["norm", "e0"], [torch.zeros(1), torch.zeros(1)])

    def test_unstamped_sources_are_never_checked(self):
        e = self._engine(None)
        e._validate_held_yields(0, ["anything"], [None])


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
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import rdt_send

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
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import rdt_send

        pm, seen = self._fake_modules(monkeypatch)
        with rdt_send._qkv_index_device_ctx():
            pm.split_qkv_weights(None, torch.empty(2, 2))
        assert seen == ["cpu"]

    def test_originals_and_torch_arange_are_restored(self, monkeypatch):
        """torch.arange is patched process-wide for the duration of ONE call, so a
        leak would silently put every later index tensor on a device."""
        import torch

        pytest.importorskip("megatron.bridge.models.conversion.param_mapping")
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import rdt_send

        pm, _seen = self._fake_modules(monkeypatch)
        before, real_arange = pm.split_qkv_weights, torch.arange
        with rdt_send._qkv_index_device_ctx():
            assert pm.split_qkv_weights is not before  # wrapped
            pm.split_qkv_weights(None, torch.empty(2, 2, device="meta"))
            assert torch.arange is real_arange, "arange must be restored after each call"
        assert pm.split_qkv_weights is before
        assert torch.arange is real_arange
        assert torch.arange(3).device.type == "cpu"


@pytest.mark.vllm
class TestShardedRdtVllmRegistration:
    """The factory registration (requires the vLLM wheel)."""

    def test_engine_registered(self):
        pytest.importorskip("vllm")
        # Importing the weight_sync package's register module runs ensure_registered().
        from vllm.config import WeightTransferConfig
        from vllm.distributed.weight_transfer import WeightTransferEngineFactory

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import rdt_vllm_register

        rdt_vllm_register.ensure_registered()
        assert "sharded_rdt" in WeightTransferEngineFactory._registry
        # vLLM 0.23.0 already accepts arbitrary backend strings (Literal | str);
        # no runtime relaxation needed, and the built-ins still validate.
        assert WeightTransferConfig(backend="sharded_rdt").backend == "sharded_rdt"
        assert WeightTransferConfig(backend="nccl").backend == "nccl"
        assert WeightTransferConfig(backend="ipc").backend == "ipc"

    def test_the_registered_module_path_actually_resolves(self):
        """`register_engine` stores the engine's module path as a string and imports
        it only when a worker constructs the backend, so a stale path fails on a real
        inference worker rather than here. Force the loader to resolve it."""
        pytest.importorskip("vllm")
        from vllm.distributed.weight_transfer import WeightTransferEngineFactory

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import rdt_vllm_register
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_engine import (
            ShardedRDTWeightTransferEngine,
        )

        rdt_vllm_register.ensure_registered()
        loader = WeightTransferEngineFactory._registry["sharded_rdt"]
        assert loader() is ShardedRDTWeightTransferEngine


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

    def test_delta_create_init_info(self):
        ie_cfg = self._make_ie_cfg(weight_sync_backend="delta", run_engines_locally=False)
        # delta_weight_sync defaults to None, so a delta run must supply the whole sub-config.
        ie_cfg.delta_weight_sync = DeltaWeightSyncConfig(
            sync_dir="gs://bucket/prefix",
            local_checkpoint_dir="/tmp/receiver",
            max_file_size_in_gb=2,
            publish_num_workers=3,
            checkpoint_load_format="vllm_multi_thread_safetensors",
            multi_thread_safetensors_max_workers=4,
        )

        init_info = DeltaTransferStrategy.create_init_info(
            ie_cfg,
            base_model_path="Qwen/Qwen2.5-1.5B-Instruct",
        )

        assert isinstance(init_info, DeltaInitInfo)
        assert init_info.sync_dir == "gs://bucket/prefix"
        assert init_info.base_model_path == "Qwen/Qwen2.5-1.5B-Instruct"
        assert init_info.local_checkpoint_dir == "/tmp/receiver"
        assert init_info.checkpoint_load_format == "vllm_multi_thread_safetensors"
        assert init_info.multi_thread_safetensors_max_workers == 4
        assert init_info.max_file_size_in_gb == 2
        assert init_info.publish_num_workers == 3
        assert init_info.override_existing_receiver is True

    def test_delta_create_init_info_requires_sync_dir(self):
        ie_cfg = self._make_ie_cfg(weight_sync_backend="delta")

        with pytest.raises(ValueError, match="sync_dir"):
            DeltaTransferStrategy.create_init_info(ie_cfg, base_model_path="model")


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
