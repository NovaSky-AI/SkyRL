# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`GroupedWeightSource` group-contract tests for the sharded-RDT base.

`groups()` / `iter_groups()` define what a *group index* means -- the unit
`held_names()` narrows and that the RDT trainer gathers, publishes and frees by.
A source whose batches disagree with the trainer's own group partition does not
return wrong data, it deadlocks the ranks sharing a gather collective, so the
contract is pinned here.

Adapted from the vLLM RDT fork (`tests/distributed/test_weight_transfer.py`).
Marked `vllm`: `sharded_rdt_base` imports the base ABCs from the wheel.
"""

import importlib
from functools import cached_property
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm", reason="sharded_rdt_base imports vLLM's trainer-side ABCs")

pytestmark = pytest.mark.vllm

from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.sharded_rdt_base import (  # noqa: E402
    GroupedWeightSource,
    ParamMeta,
    layerwise_groups,
)


class TestGroupedWeightSourceContract:
    """`groups()` / `iter_groups()` on the GroupedWeightSource ABC. Group indices are
    what backends gather and free by, and `held_names()` is what narrows them,
    so the default must agree with `layerwise_groups` over `metadata`."""

    class _Source(GroupedWeightSource):
        """Minimal source over an ordered (name, tensor) list, optionally owning
        only some groups (in which case it iterates only those, per contract)."""

        def __init__(self, names, owned=None, reverse=False):
            self._pairs = [(n, torch.full((2,), float(i))) for i, n in enumerate(names)]
            self._owned = owned
            self._reverse = reverse
            self._held = None
            if owned is not None:
                all_groups = layerwise_groups(names)
                self._held = [n for i in sorted(set(owned)) for n in all_groups[i]]

        @cached_property
        def metadata(self):
            return [ParamMeta(n, t.dtype, tuple(t.shape)) for n, t in self._pairs]

        def held_names(self):
            return self._held

        def iter_full(self):
            pairs = self._pairs
            if self._owned is not None:
                all_groups = layerwise_groups([n for n, _ in pairs])
                keep = {n for i in self._owned for n in all_groups[i]}
                pairs = [(n, t) for n, t in pairs if n in keep]
            return iter(list(reversed(pairs)) if self._reverse else pairs)

    def _source(self, names, owned=None, reverse=False):
        return self._Source(names, owned, reverse)

    def test_groups_defaults_to_the_layerwise_partition(self):
        names = ["embed.w", "model.layers.0.a", "model.layers.1.a", "norm.w"]
        assert self._source(names).groups() == layerwise_groups(names)

    def test_groups_is_restricted_to_owned_groups(self):
        names = ["embed.w", "model.layers.0.a", "model.layers.1.a", "norm.w"]
        assert self._source(names, owned=[1, 2]).groups() == [
            ["model.layers.0.a"],
            ["model.layers.1.a"],
        ]

    def test_groups_follows_the_partition_order_not_the_declaration_order(self):
        """``groups()`` filters the partition, so held names declared in any order
        still pair with the right batch of ``iter_groups()``."""
        names = ["embed.w", "model.layers.0.a", "model.layers.1.a", "norm.w"]
        source = self._source(names, owned=[2, 1, 2])
        assert source.groups() == [["model.layers.0.a"], ["model.layers.1.a"]]
        assert [ns for ns, _ in source.iter_groups()] == [
            ["model.layers.0.a"],
            ["model.layers.1.a"],
        ]

    def test_iter_groups_batches_the_stream_per_group(self):
        names = ["embed.w", "model.layers.0.a", "model.layers.0.b", "norm.w"]
        batches = list(self._source(names).iter_groups())
        assert [ns for ns, _ in batches] == [
            ["embed.w"],
            ["model.layers.0.a", "model.layers.0.b"],
            ["norm.w"],
        ]
        assert all(len(ns) == len(ts) for ns, ts in batches)

    def test_iter_groups_yields_the_tensors_iteration_produced(self):
        names = ["model.layers.0.a", "model.layers.0.b"]
        (batch,) = list(self._source(names).iter_groups())
        _names, tensors = batch
        assert [float(t[0]) for t in tensors] == [0.0, 1.0]

    def test_iter_groups_yields_only_owned_groups(self):
        names = ["embed.w", "model.layers.0.a", "model.layers.1.a"]
        batches = list(self._source(names, owned=[2]).iter_groups())
        assert [ns for ns, _ in batches] == [["model.layers.1.a"]]

    def test_out_of_order_iteration_raises(self):
        """Materializing is usually a collective, so a rank that iterates out of
        order deadlocks its peers -- fail loudly instead."""
        source = self._source(["model.layers.0.a", "model.layers.0.b"], reverse=True)
        with pytest.raises(RuntimeError, match="iteration order must match"):
            list(source.iter_groups())

    def test_a_source_may_override_iter_groups(self):
        """The extension point: materialize a whole group in one step instead of
        one generator resume per tensor."""
        calls = []
        base = self._Source

        class _Batched(base):
            def iter_groups(self):
                for group in self.groups():
                    calls.append(len(group))
                    yield group, [torch.zeros(2) for _ in group]

        source = _Batched(["model.layers.0.a", "model.layers.0.b"])
        assert [ns for ns, _ in source.iter_groups()] == [["model.layers.0.a", "model.layers.0.b"]]
        assert calls == [2]


class TestHeldNamesDefault:
    """`held_names()` is a declaration with a safe default: a source that never
    overrides it holds the whole model, which the engine reads as every producer
    owning every name."""

    def test_the_default_is_none(self):
        names = ["embed.w", "model.layers.0.a"]
        src = TestGroupedWeightSourceContract._Source(names)
        assert src.held_names() is None


class TestRdtFsdpWeightSource:
    """RDT's FSDP source is the shared one re-ordered **group-major**.

    That reorder is not cosmetic: ``layerwise_groups`` must partition
    ``metadata`` exactly, because that partition IS the group index the
    consumers' pull plans and the producer's free barrier are keyed on. A
    ``state_dict()`` whose layers interleave with the pre/post block would
    otherwise produce groups that are not contiguous runs of the metadata.
    """

    @staticmethod
    def _model(names):
        model = torch.nn.Module()
        for name in names:
            # register_parameter rejects dots, so nest a holder module per name.
            holder = model
            parts = name.split(".")
            for part in parts[:-1]:
                child = getattr(holder, part, None)
                if child is None:
                    child = torch.nn.Module()
                    holder.add_module(part, child)
                holder = child
            holder.register_parameter(parts[-1], torch.nn.Parameter(torch.ones(2, 2)))
        return model

    def _source(self, names):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            make_fsdp_weight_source,
        )

        return make_fsdp_weight_source(self._model(names), torch.bfloat16)

    # state_dict order with the layers out of numeric order, which is what the
    # reorder actually has to fix.
    _INTERLEAVED = [
        "model.embed_tokens.weight",
        "model.layers.1.a",
        "model.layers.0.a",
        "lm_head.weight",
    ]

    def test_reorders_into_group_major_order(self):
        src = self._source(self._INTERLEAVED)
        assert [m.name for m in src.metadata] == [
            "model.embed_tokens.weight",
            "model.layers.0.a",
            "model.layers.1.a",
            "lm_head.weight",
        ]

    def test_groups_partition_metadata_exactly(self):
        src = self._source(self._INTERLEAVED)
        flat = [n for g in src.groups() for n in g]
        assert flat == [m.name for m in src.metadata]
        assert src.groups() == [
            ["model.embed_tokens.weight"],
            ["model.layers.0.a"],
            ["model.layers.1.a"],
            ["lm_head.weight"],
        ]

    def test_unindexed_names_split_by_position_not_by_role(self):
        """``layerwise_groups`` puts un-indexed names before the first indexed
        one in ``pre`` and the rest in ``post`` — it does not know what an
        embedding is. So a state_dict that emitted a layer before the embedding
        would file the embedding under ``post``. Real ``state_dict()`` order is
        definition order, so this does not arise; it is pinned because the
        partition is what the consumers' pull plans are keyed on, and getting it
        silently different per rank is the failure mode that matters."""
        src = self._source(["model.layers.0.a", "model.embed_tokens.weight"])
        assert src.groups() == [["model.layers.0.a"], ["model.embed_tokens.weight"]]

    def test_channels_agree_after_the_reorder(self):
        src = self._source(self._INTERLEAVED)
        meta = src.metadata
        pairs = list(src.iter_full())
        assert [m.name for m in meta] == [n for n, _ in pairs]
        assert [m.shape for m in meta] == [tuple(t.shape) for _, t in pairs]
        assert [m.dtype for m in meta] == [t.dtype for _, t in pairs]

    def test_iter_groups_batches_per_layer(self):
        src = self._source(["model.layers.0.a", "model.layers.0.b", "model.layers.1.a"])
        assert [names for names, _ in src.iter_groups()] == [
            ["model.layers.0.a", "model.layers.0.b"],
            ["model.layers.1.a"],
        ]

    def test_holds_everything_by_default(self):
        """FSDP replicates the whole model on every rank after the gather, so
        there is no ownership to declare — unlike the PP/EP-local Megatron
        source."""
        assert self._source(["model.layers.0.a"]).held_names() is None


class TestRdtMegatronWeightSource:
    """The whole-model Megatron fallback: the shared source plus RDT's inherited
    group channels, and nothing else.

    Worth a test of its own because it is a diamond — ``MegatronWeightSource``
    and ``GroupedWeightSource`` both derive from vLLM's ``WeightSource``, and
    ``GroupedWeightSource`` re-declares ``metadata`` / ``__iter__`` abstract. If
    the MRO put the abstract declarations first the class would not instantiate
    at all.
    """

    class _Bridge:
        def __init__(self, names):
            self._names = names

        def export_hf_weights(self, module, show_progress=False, conversion_tasks=None):
            assert conversion_tasks is None, "the fallback exports the whole model in one call"
            return ((n, torch.ones(2, 2)) for n in self._names)

    def _source(self, names):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            RdtMegatronWeightSource,
        )

        return RdtMegatronWeightSource(self._Bridge(names), object(), torch.bfloat16)

    def test_instantiates_and_streams(self):
        src = self._source(["model.embed_tokens.weight", "model.layers.0.a"])
        assert [m.name for m in src.metadata] == ["model.embed_tokens.weight", "model.layers.0.a"]
        assert [n for n, _ in src.iter_full()] == ["model.embed_tokens.weight", "model.layers.0.a"]

    def test_groups_partition_the_bridges_canonical_order(self):
        """The bridge yields HF-canonical order, which is already
        group-contiguous, so no reorder is needed here."""
        src = self._source(["model.embed_tokens.weight", "model.layers.0.a", "model.layers.1.a", "lm_head.weight"])
        assert src.groups() == [
            ["model.embed_tokens.weight"],
            ["model.layers.0.a"],
            ["model.layers.1.a"],
            ["lm_head.weight"],
        ]

    def test_holds_everything(self):
        """Whole-model residency is the point for a pull backend: each producer
        must be able to serve its bound consumer the complete model."""
        assert self._source(["model.layers.0.a"]).held_names() is None


class TestExpertNameResolution:
    """Expert HF names come from the bridge's mapping registry, never an assumed
    layout: architectures differ (Kimi K2.5-VL nests its decoder stack under
    `language_model.`, Qwen3-MoE does not), and a wrong name is never baked by the
    consumer, so those experts silently keep stale weights instead of raising.

    `megatron_to_hf_lookup` is pure string work — no model, no CUDA, no collective —
    which is what lets this be a CPU test.
    """

    @pytest.mark.megatron
    @pytest.mark.parametrize("projection", ["fc1", "fc2"])
    def test_lora_expert_template_resolves_in_qwen_registry(self, projection):
        from megatron.bridge.models.qwen.qwen3_moe_bridge import Qwen3MoEBridge

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            MegatronStackedWeightSource,
        )

        sample = f"decoder.layers.3.mlp.experts.linear_{projection}.to_wrap.weight5"
        name = MegatronStackedWeightSource._mg_expert_template(sample).format(layer=47, e=11)
        mapping = Qwen3MoEBridge.mapping_registry(None).megatron_to_hf_lookup(name)

        base = "model.layers.47.mlp.experts.11"
        expected = {"gate": f"{base}.gate_proj.weight", "up": f"{base}.up_proj.weight"}
        assert mapping is not None
        assert mapping.hf_param == (expected if projection == "fc1" else f"{base}.down_proj.weight")

    def test_template_resubstitutes_both_indices(self):
        """Only the SHAPE of the sample name is kept — a foreign expert this rank
        holds no task for still gets the right name."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            MegatronStackedWeightSource as S,
        )

        t = S._mg_expert_template("decoder.layers.3.mlp.experts.linear_fc1.weight5")
        assert t.format(layer=7, e=11) == "decoder.layers.7.mlp.experts.linear_fc1.weight11"

    def test_template_keeps_a_nested_stack_prefix(self):
        """A nested stack prefix must survive the resubstitution."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            MegatronStackedWeightSource as S,
        )

        t = S._mg_expert_template("language_model.decoder.layers.3.mlp.experts.linear_fc2.weight5")
        assert t.format(layer=0, e=2) == "language_model.decoder.layers.0.mlp.experts.linear_fc2.weight2"

    @pytest.mark.megatron
    @pytest.mark.parametrize(
        "module,cls_hint,mg_prefix,hf_prefix",
        [
            ("megatron.bridge.models.qwen.qwen3_moe_bridge", "Bridge", "decoder", "model"),
            (
                "megatron.bridge.models.kimi_vl.kimi_k25_vl_bridge",
                "Bridge",
                "language_model.decoder",
                "language_model.model",
            ),
        ],
    )
    def test_registry_resolves_expert_names(self, module, cls_hint, mg_prefix, hf_prefix):
        """The registry returns CONCRETE names with both indices substituted, and
        names gate/up by KEY rather than by position. Qwen3-MoE resolves under
        `model.`, Kimi under `language_model.`.
        """
        import importlib
        import inspect

        mod = importlib.import_module(module)
        bridge_cls = next(
            o for n, o in vars(mod).items() if inspect.isclass(o) and cls_hint in n and o.__module__ == mod.__name__
        )
        # `mapping_registry` never touches `self`, so no model is required.
        reg = bridge_cls.mapping_registry(None)

        layer, expert = 3, 5
        fc1 = reg.megatron_to_hf_lookup(f"{mg_prefix}.layers.{layer}.mlp.experts.linear_fc1.weight{expert}").hf_param
        fc2 = reg.megatron_to_hf_lookup(f"{mg_prefix}.layers.{layer}.mlp.experts.linear_fc2.weight{expert}").hf_param

        base = f"{hf_prefix}.layers.{layer}.mlp.experts.{expert}"
        assert fc1 == {"gate": f"{base}.gate_proj.weight", "up": f"{base}.up_proj.weight"}
        assert fc2 == f"{base}.down_proj.weight"


class TestMtpExperts:
    """A trained MTP head's experts are their own group, under the HF layer the bridge maps them to."""

    @staticmethod
    def _task(name):
        from types import SimpleNamespace

        return SimpleNamespace(param_name=name, global_param_name=name)

    def test_mtp_experts_are_keyed_apart_from_decoder_layer_0(self):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            MegatronStackedWeightSource,
        )

        tasks = [
            self._task("decoder.layers.0.mlp.experts.linear_fc1.weight0"),
            self._task("mtp.layers.0.transformer_layer.mlp.experts.linear_fc1.weight0"),
            self._task("mtp.layers.0.eh_proj.weight"),
        ]
        src = MegatronStackedWeightSource.__new__(MegatronStackedWeightSource)
        non_expert, layers = src._partition_tasks(tasks)
        assert [t.global_param_name for t in non_expert] == ["mtp.layers.0.eh_proj.weight"]
        assert sorted(layers) == [("decoder.", 0), ("mtp.", 0)]

    @staticmethod
    def _source_with_registry(reg):
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            MegatronStackedWeightSource,
        )

        model_bridge = SimpleNamespace(mapping_registry=lambda: reg)
        bridge = SimpleNamespace(_model_bridge=model_bridge, hf_pretrained=None)
        return MegatronStackedWeightSource(bridge, module=None, dtype=torch.bfloat16)

    @staticmethod
    def _mg_names(n_layers, n_mtp, moe, hybrid=False):
        """Megatron names of a tiny model: a few decoder layers, then ``n_mtp`` MTP layers.

        ``hybrid``: a HybridModel (GLM-5 under megatron-bridge 0.8), whose decoder layer ``i`` is attention layer
        ``2i`` and MLP layer ``2i + 1``.
        """

        def block(attn, mlp):
            if not moe:
                return [f"{attn}.self_attention.linear_proj.weight", f"{mlp}.mlp.linear_fc2.weight"]
            experts = [f"{mlp}.mlp.experts.linear_fc{k}.weight{e}" for k in (1, 2) for e in range(2)]
            return [f"{attn}.self_attention.linear_proj.weight", f"{mlp}.mlp.router.weight", *experts]

        def layer(prefix, i):
            if hybrid:
                return block(f"{prefix}.{2 * i}", f"{prefix}.{2 * i + 1}")
            return block(f"{prefix}.{i}", f"{prefix}.{i}")

        final_norm = "decoder.final_norm.weight" if hybrid else "decoder.final_layernorm.weight"
        names = ["embedding.word_embeddings.weight", final_norm, "output_layer.weight"]
        names += [n for i in range(n_layers) for n in layer("decoder.layers", i)]
        for i in range(n_mtp):
            names += [f"mtp.layers.{i}.{p}.weight" for p in ("enorm", "hnorm", "eh_proj", "final_layernorm")]
            if hybrid:
                names += layer(f"mtp.layers.{i}.mtp_model_layer.layers", 0)
            else:
                names += block(f"mtp.layers.{i}.mtp_model_layer", f"mtp.layers.{i}.mtp_model_layer")
        return names

    @pytest.mark.megatron
    @pytest.mark.parametrize(
        "bridge, n_mtp, moe, hybrid",
        [
            ("glm_moe_dsa.glm5_bridge.GLM5Bridge", 1, True, True),  # the head is model.layers.<num_hidden_layers>
            ("deepseek.deepseek_v3_bridge.DeepSeekV3Bridge", 1, True, False),
            ("qwen.qwen3_next_bridge.Qwen3NextBridge", 1, True, False),  # mtp.layers.0, and un-indexed mtp.fc etc.
            ("mimo.mimo_bridge.MimoBridge", 2, False, False),  # model.mtp_layers.<i>: no ".layers."
        ],
    )
    def test_the_walk_is_the_partition_for_each_bridges_mtp_names(self, bridge, n_mtp, moe, hybrid):
        """Through the real mapping registry, at pp=1: the walk's batches are ``layerwise_groups`` over its
        metadata, head included, and every name the tasks map to is sent once, with no layer's experts
        keyed onto another's."""
        pytest.importorskip("megatron.bridge", reason="needs the megatron extra")
        from megatron.bridge.models.conversion.utils import extract_sort_key

        module, cls = bridge.rsplit(".", 1)
        bridge_cls = getattr(importlib.import_module(f"megatron.bridge.models.{module}"), cls)
        bridge_obj = bridge_cls.__new__(bridge_cls)  # mapping_registry reads only hf_config
        bridge_obj.hf_config = SimpleNamespace(num_hidden_layers=3, num_nextn_predict_layers=n_mtp)
        reg = bridge_obj.mapping_registry()

        def hf_names(task):
            hf = reg.megatron_to_hf_lookup(task.global_param_name).hf_param
            return list(hf.values()) if isinstance(hf, dict) else [hf]

        # In the bridge's own task order, which streams mtp.layers.<i> with decoder layer i.
        mg_names = sorted(self._mg_names(3, n_mtp, moe, hybrid), key=extract_sort_key)
        tasks = [SimpleNamespace(param_name=n, global_param_name=n, param_weight=torch.zeros(4, 2)) for n in mg_names]
        src = self._source_with_registry(reg)
        src._bridge.get_conversion_tasks = lambda module: tasks
        src._bridge._model_bridge.stream_weights_megatron_to_hf = lambda *args, conversion_tasks, **kwargs: (
            (name, torch.zeros(2)) for t in conversion_tasks for name in hf_names(t)
        )

        meta = [m.name for m in src.metadata]
        assert [names for names, _ in src.iter_groups()] == layerwise_groups(meta)
        assert sorted(meta) == sorted(name for t in tasks for name in hf_names(t))

    def test_a_dense_model_with_an_mtp_head_gets_the_stacked_source(self, monkeypatch):
        """The plain source streams the bridge's order, where ``mtp.layers.0`` sits with decoder layer 0: not
        group-contiguous. A dense model with an MTP head therefore gets the stacked source, which reorders it."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import rdt_send

        class _Stacked:
            _EXPERT_PRED = rdt_send.MegatronStackedWeightSource._EXPERT_PRED
            _demoted = False

            def __init__(self, bridge, module, dtype):
                pass

            def held_names(self):
                return None

        monkeypatch.setattr(rdt_send, "MegatronStackedWeightSource", _Stacked)
        tasks = [
            self._task("decoder.layers.0.self_attention.linear_proj.weight"),
            self._task("mtp.layers.0.eh_proj.weight"),
        ]
        for t in tasks:
            t.mapping = SimpleNamespace()
        bridge = SimpleNamespace(get_conversion_tasks=lambda module: tasks)
        assert isinstance(rdt_send.make_megatron_weight_source(bridge, None, torch.bfloat16), _Stacked)
        tasks.pop()  # no MTP head: a dense model at pp == 1 keeps the plain source
        assert isinstance(
            rdt_send.make_megatron_weight_source(bridge, None, torch.bfloat16), rdt_send.RdtMegatronWeightSource
        )

    def test_lora_merges_each_layers_own_adapters(self):
        """Adapters are keyed by the wrapped linear's Megatron name: an MTP head's are not decoder layer 0's."""
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            MegatronStackedWeightSource,
            _ExpertLayer,
        )

        head = "mtp.layers.0.transformer_layer.mlp.experts"
        fc1, fc2 = ([self._task(f"{head}.linear_fc{k}.to_wrap.weight0")] for k in (1, 2))
        lay = _ExpertLayer(
            megatron_index=0,
            checkpoint_key=("model.layers", 4),
            fc1=fc1,
            fc2=fc2,
            owned=True,
            n_local=1,
            F=1,
            H=1,
            ep_size=1,
        )

        def adapter(rows, scale):  # B @ A is `scale` everywhere
            lin_in = SimpleNamespace(param_weight=torch.ones(1, 1))
            lin_out = SimpleNamespace(param_weight=torch.full((rows, 1), scale))
            return [SimpleNamespace(linear_in_task=lin_in, linear_out_task=lin_out, alpha=1, dim=1)]

        tasks_by_base = {
            f"{prefix}.linear_fc{k}": adapter(rows, scale)
            for prefix, scale in (("decoder.layers.0.mlp.experts", 1.0), (head, 2.0))
            for k, rows in ((1, 2), (2, 1))
        }
        fc1_local, fc2_local = torch.zeros(1, 2, 1), torch.zeros(1, 1, 1)
        src = MegatronStackedWeightSource.__new__(MegatronStackedWeightSource)
        src._merge_lora_into_local_shards(lay, fc1_local, fc2_local, tasks_by_base)
        assert fc1_local.eq(2).all() and fc2_local.eq(2).all()

    def test_mtp_tasks_stream_after_the_decoder_layers(self, monkeypatch):
        """The bridge sorts ``mtp.layers.0`` with decoder layer 0; the plan moves it behind the decoder."""
        from types import SimpleNamespace

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            MegatronStackedWeightSource,
        )

        tasks = [
            self._task("output_layer.weight"),
            self._task("mtp.layers.0.eh_proj.weight"),
            self._task("decoder.layers.0.input_layernorm.weight"),
            self._task("mtp.layers.0.enorm.weight"),
            self._task("decoder.layers.1.input_layernorm.weight"),
        ]
        src = MegatronStackedWeightSource.__new__(MegatronStackedWeightSource)
        src._bridge, src._module = SimpleNamespace(get_conversion_tasks=lambda module: tasks), object()
        monkeypatch.setattr(src, "_expert_geometry", lambda raw_layers: raw_layers, raising=False)

        non_expert, _, _ = src._build_export_plan()

        assert [t.global_param_name for t in non_expert] == [
            "output_layer.weight",
            "decoder.layers.0.input_layernorm.weight",
            "decoder.layers.1.input_layernorm.weight",
            "mtp.layers.0.eh_proj.weight",
            "mtp.layers.0.enorm.weight",
        ]

    @staticmethod
    def _last_stage_walk(monkeypatch, stream_layers, tail=()):
        """The PP-local walk of a last stage holding decoder layers 2 and 3 and the MTP head (HF layer 4).

        Each layer's local experts hold the layer's index, and the export ring is on at depth 2. The
        bridge streams the post block first, then ``stream_layers`` in order, then the un-indexed ``tail``.
        """
        import contextlib
        from types import SimpleNamespace

        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt import rdt_send
        from skyrl.backends.skyrl_train.weight_sync.sharded_rdt.rdt_send import (
            MegatronStackedWeightSource,
            _ExpertLayer,
        )

        F, H = 4, 3

        def lay(layer):
            fc1 = [SimpleNamespace(param_weight=torch.full((2 * F, H), float(layer)))]
            fc2 = [SimpleNamespace(param_weight=torch.full((H, F), float(layer)))]
            key = ("model.layers", layer)
            return _ExpertLayer(
                megatron_index=layer, checkpoint_key=key, fc1=fc1, fc2=fc2, owned=True, n_local=1, F=F, H=H, ep_size=1
            )

        stream = [("model.norm.weight", torch.zeros(H)), ("lm_head.weight", torch.zeros(H))]
        stream += [(f"model.layers.{layer}.input_layernorm.weight", torch.zeros(H)) for layer in stream_layers]
        stream += [(name, torch.zeros(H)) for name in tail]
        src = MegatronStackedWeightSource.__new__(MegatronStackedWeightSource)
        src._dtype, src._pp_local, src._demoted = torch.float32, True, False
        src._pp_geometry = lambda: (2, 1)
        src._stack_ring, src._stack_ring_pos, src._stack_ring_depth, src._stack_ring_on = {}, 0, 2, True
        src._expert_names, src._layer_geom, src._my_ep_rank = {}, {}, 0
        src._bridge, src._module = SimpleNamespace(hf_pretrained=None), None
        src._build_export_plan = lambda: ([], {("model.layers", layer): lay(layer) for layer in (2, 3, 4)}, None)
        stream_fn = lambda *args, **kwargs: iter(stream)  # noqa: E731
        src._prepared_model_bridge = lambda: SimpleNamespace(stream_weights_megatron_to_hf=stream_fn)
        src._expert_names_for = lambda lay, E: [
            f"model.layers.{lay.checkpoint_key[1]}.mlp.experts.0.{p}_proj.weight" for p in ("gate", "up", "down")
        ]
        # The partition: decoder layers, the MTP head, then the post block.
        groups = [[f"model.layers.{layer}.input_layernorm.weight"] for layer in (2, 3, 4)]
        groups.append(["model.norm.weight", "lm_head.weight", *tail])
        src._group_index_of_name = {n: gi for gi, g in enumerate(groups) for n in g}
        src._owned_group_idx = list(range(len(groups)))
        monkeypatch.setattr(rdt_send, "_pp_local_export_ctx", contextlib.nullcontext)
        monkeypatch.setattr(rdt_send, "_qkv_index_device_ctx", contextlib.nullcontext)
        return [
            (
                [n for n in names if ".experts." not in n],
                {float(t.flatten()[0]) for n, t in zip(names, tensors) if ".experts." in n},
            )
            for names, tensors in src._iter_groups_impl(collect_meta=False)
        ]

    def test_each_layer_group_carries_its_own_experts(self, monkeypatch):
        groups = self._last_stage_walk(monkeypatch, [2, 3, 4])
        assert groups == [
            (["model.layers.2.input_layernorm.weight"], {2.0}),
            (["model.layers.3.input_layernorm.weight"], {3.0}),
            (["model.layers.4.input_layernorm.weight"], {4.0}),
            (["model.norm.weight", "lm_head.weight"], set()),
        ]

    def test_a_later_stage_sends_all_its_un_indexed_names_as_the_post_group(self, monkeypatch):
        """On both sides of its layers: the output block before them, an MTP head's own norm after."""
        groups = self._last_stage_walk(monkeypatch, [2, 3, 4], tail=["mtp.norm.weight"])
        assert groups[3:] == [(["model.norm.weight", "lm_head.weight", "mtp.norm.weight"], set())]

    def test_a_layer_group_held_behind_later_layers_raises(self, monkeypatch):
        """Its experts would be sent from export-ring slots the later layers have already reused."""
        with pytest.raises(RuntimeError, match="overwritten before they are sent"):
            self._last_stage_walk(monkeypatch, [4, 2, 3])
