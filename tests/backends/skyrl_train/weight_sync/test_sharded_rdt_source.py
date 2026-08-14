# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""`WeightSource` group-contract tests for the vendored sharded-RDT base.

`groups()` / `iter_groups()` define what a *group index* means -- the unit
`owned_groups()` names and that the RDT trainer gathers, publishes and frees by.
A source whose batches disagree with the trainer's own group partition does not
return wrong data, it deadlocks the ranks sharing a gather collective, so the
contract is pinned here.

Vendored alongside `sharded_rdt_base.py` from the vLLM RDT fork
(`tests/distributed/test_weight_transfer.py`). Needs no vLLM: torch only.
"""

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync.sharded_rdt_base import (
    ParamMeta,
    WeightSource,
    layerwise_groups,
)


class TestWeightSourceGroupContract:
    """`groups()` / `iter_groups()` on the WeightSource ABC. Group indices are
    what backends gather and free by, and `held_names()` is what narrows them,
    so the default must agree with `layerwise_groups` over `metadata()`."""

    class _Source(WeightSource):
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
        src = TestWeightSourceGroupContract._Source(names)
        assert src.held_names() is None
