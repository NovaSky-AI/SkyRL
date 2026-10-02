"""Checkpoint mappings must refer to the same parameter, even for equal shapes."""

import sys
from types import ModuleType

import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron.patch_optimizer_group_order import (
    make_fp32_first_group_ranges,
    patch_optimizer_group_order,
)


@pytest.mark.parametrize(
    "dtypes",
    [
        [torch.bfloat16, torch.float32, torch.bfloat16, torch.float32],
        [torch.float16, torch.float32, torch.bfloat16],
        [torch.float32, torch.bfloat16, torch.float32],
        [torch.bfloat16, torch.bfloat16],
        [torch.float32, torch.float32],
        [],
    ],
)
def test_index_map_matches_optimizer_shard_order_and_keeps_group_metadata(dtypes):
    # Equal-sized tensors expose identity mistakes that shape assertions miss.
    params = [torch.nn.Parameter(torch.full((4,), i + 1.0, dtype=dtype)) for i, dtype in enumerate(dtypes)]
    other = torch.nn.Parameter(torch.ones(4, dtype=torch.float32))
    originals = [{"params": params.copy(), "lr": 0.1}, {"params": [other], "lr": 0.2}]
    groups = [{"params": params.copy(), "orig_group": originals[0]}, {"params": [other], "orig_group": originals[1]}]
    mapping = {p: (g, i) for g, group in enumerate(groups) for i, p in enumerate(group["params"])}
    sentinel = object()

    def original(cls, param_groups, buffers):
        assert cls is sentinel and param_groups is originals and buffers == ["buffer order"]
        return mapping, groups

    actual_map, actual_groups = make_fp32_first_group_ranges(original)(sentinel, originals, ["buffer order"])
    expected = [p for p in params if p.dtype == torch.float32] + [p for p in params if p.dtype != torch.float32]
    assert actual_map is mapping and actual_groups is groups
    assert [id(p) for p in groups[0]["params"]] == [id(p) for p in expected]
    for i, param in enumerate(expected):
        assert mapping[param] == (0, i)
    assert mapping[other] == (1, 0)
    assert [id(p) for p in originals[0]["params"]] == [id(p) for p in params]
    assert groups[0]["orig_group"] is originals[0]
    assert groups[1]["orig_group"]["lr"] == 0.2
    for i, param in enumerate(params):
        torch.testing.assert_close(param, torch.full_like(param, i + 1.0))


def test_installer_is_idempotent_and_preserves_classmethod(monkeypatch):
    module = ModuleType("megatron.core.optimizer.distrib_optimizer")

    class Optimizer:
        @classmethod
        def _build_optimizer_group_ranges(cls, param_groups, gbuf_ranges):
            assert cls is Optimizer
            return {}, []

    module.DistributedOptimizer = Optimizer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    patch_optimizer_group_order()
    installed = Optimizer._build_optimizer_group_ranges.__func__
    patch_optimizer_group_order()
    assert Optimizer._build_optimizer_group_ranges.__func__ is installed
    assert Optimizer._build_optimizer_group_ranges([], []) == ({}, [])
