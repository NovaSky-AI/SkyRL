# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`GroupedWeightSource` group-contract tests for the sharded-RDT base.

`groups()` / `iter_groups()` define what a *group index* means -- the unit
`held_names()` narrows and that the RDT trainer gathers, publishes and frees by.
A source whose batches disagree with the trainer's own group partition does not
return wrong data, it deadlocks the ranks sharing a gather collective, so the
contract is pinned here.

Adapted from the vLLM RDT fork (`tests/distributed/test_weight_transfer.py`).
Marked `vllm` because `sharded_rdt_base` now imports the base ABCs from the
wheel rather than vendoring them.
"""

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
    so the default must agree with `layerwise_groups` over `metadata()`."""

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

        def metadata(self):
            return [ParamMeta(n, t.dtype, tuple(t.shape)) for n, t in self._pairs]

        def held_names(self):
            return self._held

        def __iter__(self):
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
    ``metadata()`` exactly, because that partition IS the group index the
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
        assert [m.name for m in src.metadata()] == [
            "model.embed_tokens.weight",
            "model.layers.0.a",
            "model.layers.1.a",
            "lm_head.weight",
        ]

    def test_groups_partition_metadata_exactly(self):
        src = self._source(self._INTERLEAVED)
        flat = [n for g in src.groups() for n in g]
        assert flat == [m.name for m in src.metadata()]
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
        meta = src.metadata()
        pairs = list(src)
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
        assert [m.name for m in src.metadata()] == ["model.embed_tokens.weight", "model.layers.0.a"]
        assert [n for n, _ in src] == ["model.embed_tokens.weight", "model.layers.0.a"]

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

        pytest.importorskip("megatron.bridge", reason="needs the megatron extra")
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
