"""Ownership and gather-group tests for the sharded-RDT weight sources.

The code under test is the per-rank ownership under PP/EP in
``sharded_rdt/rdt_send.py``, which is what keeps a PP-local export from
materializing the whole model on every rank.

``held_names()`` is the whole ownership contract: it declares which parameters a
rank holds, consumers route pulls by it, and the producer's free barrier counts
groups derived from it. The default (hold everything) is correct at pp=1/ep=1 and
wrong above it, which is exactly the kind of bug that shows up as a hang rather
than an error.

Every test imports inside the method, so this module needs no vLLM or Megatron at
collection time.
"""

import pytest
import torch

# Import skyrl before anything can reach megatron-bridge: ``skyrl/__init__.py``
# runs ``disable_flash_attn_cute()``, which poisons ``sys.modules["flash_attn.cute"]``
# so megatron-core's FA4 probe sees an ImportError instead of the AttributeError
# flash-attn 2.8.3 raises against nvidia-cutlass-dsl 4.6. Every test below imports
# lazily inside the method, so without this the shim would not have run by the time
# ``_fake_modules`` imports ``param_mapping`` -- and ``pytest.importorskip`` only
# catches ImportError.
import skyrl  # noqa: F401


class TestPpLocalOwnership:
    """``MegatronStackedWeightSource`` ownership detection (the PP grain).

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
        it is a wiring bug — ``make_megatron_weight_source`` should have
        delegated to the plain ``RdtMegatronWeightSource``."""
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

        e = self._engine({"norm", "e1"})
        e._validate_held_yields(0, ["norm", "e0", "e1"], [torch.zeros(1), None, torch.zeros(1)])

    def test_a_held_name_yielding_none_raises(self):
        """The dangerous direction: served_names advertises the name, consumers
        route pulls here, and the cache wait would never complete."""

        e = self._engine({"norm", "e0"})
        with pytest.raises(RuntimeError, match="disagrees with the yielded tensors"):
            e._validate_held_yields(0, ["norm", "e0"], [torch.zeros(1), None])

    def test_a_foreign_name_yielding_a_tensor_raises(self):

        e = self._engine({"norm"})
        with pytest.raises(RuntimeError, match="disagrees with the yielded tensors"):
            e._validate_held_yields(0, ["norm", "e0"], [torch.zeros(1), torch.zeros(1)])

    def test_unstamped_sources_are_never_checked(self):
        e = self._engine(None)
        e._validate_held_yields(0, ["anything"], [None])


class TestQkvIndexDeviceCtx:
    """``_qkv_index_device_ctx`` keeps the QKV split's index tensors on the weight's
    device instead of the host, which is worth ~0.65s/sync at 235B (a CPU index
    tensor against a CUDA weight forces an H2D copy + stream sync per gather).
    It deliberately copies NO upstream logic — it only changes where
    ``torch.arange`` allocates — so these cover the wrapping, the injection, and
    the restore. A meta device stands in for CUDA so this needs no GPU."""

    @staticmethod
    def _fake_modules(monkeypatch):
        """Stub split fns on the real modules, so the context wraps something we can
        observe. Records the device each torch.arange call landed on."""
        from megatron.bridge.models.conversion import param_mapping as pm

        seen = []

        def _split(config, qkv, *a, **kw):
            seen.append(torch.arange(4).device.type)
            return ("q", "k", "v")

        for name in ("split_qkv_weights", "split_qkv_biases", "split_qkv_weights_scale"):
            monkeypatch.setattr(pm, name, _split, raising=False)
        return pm, seen

    def test_index_tensors_follow_the_weight_device(self, monkeypatch):
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
        pytest.importorskip("megatron.bridge.models.conversion.param_mapping")

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import rdt_send

        pm, seen = self._fake_modules(monkeypatch)
        with rdt_send._qkv_index_device_ctx():
            pm.split_qkv_weights(None, torch.empty(2, 2))
        assert seen == ["cpu"]

    def test_originals_and_torch_arange_are_restored(self, monkeypatch):
        """torch.arange is patched process-wide for the duration of ONE call, so a
        leak would silently put every later index tensor on a device."""
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
