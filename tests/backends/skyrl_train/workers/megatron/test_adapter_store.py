"""Unit tests for Megatron LoRA adapter CPU snapshots."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


def _load_adapter_store():
    """Load the unit under test without importing GPU-only Megatron libraries."""

    class FakeDDP:
        pass

    class FakeChainedOptimizer:
        pass

    megatron = ModuleType("megatron")
    core = ModuleType("megatron.core")
    parallel_state = ModuleType("megatron.core.parallel_state")
    distributed = ModuleType("megatron.core.distributed")
    optimizer = ModuleType("megatron.core.optimizer")
    distributed.DistributedDataParallel = FakeDDP
    optimizer.ChainedOptimizer = FakeChainedOptimizer
    core.parallel_state = parallel_state
    megatron.core = core

    stubs = {
        "megatron": megatron,
        "megatron.core": core,
        "megatron.core.parallel_state": parallel_state,
        "megatron.core.distributed": distributed,
        "megatron.core.optimizer": optimizer,
    }
    module_name = "adapter_store_under_test"
    path = (
        Path(__file__).parents[5]
        / "skyrl/backends/skyrl_train/workers/megatron/adapter_store.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    previous = {name: sys.modules.get(name) for name in stubs}
    try:
        sys.modules.update(stubs)
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
        for name, original in previous.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


adapter_store = _load_adapter_store()


def test_optimizer_none_shard_placeholders_round_trip(monkeypatch):
    """Ranks without a local optimizer shard keep their None placeholder."""
    monkeypatch.setattr(adapter_store, "_new_pinned_like", torch.empty_like)

    live_param = torch.tensor([1.0, 2.0])
    inner_optimizer = SimpleNamespace(state={}, param_groups=[{}])
    optimizer = SimpleNamespace(
        shard_fp32_from_float16_groups=[[live_param, None]],
        optimizer=inner_optimizer,
    )
    store = adapter_store.AdapterStore()

    source = store._allocate_empty_slot([], optimizer)
    store._snapshot(source, [], optimizer)
    assert source.cpu_main_param[0][0][1] is None

    live_param.zero_()
    store._restore(source, [], optimizer)
    torch.testing.assert_close(live_param, torch.tensor([1.0, 2.0]))

    destination = store._allocate_empty_slot([], optimizer)
    store._copy_slot(source, destination)
    torch.testing.assert_close(destination.cpu_main_param[0][0][0], live_param)
    assert destination.cpu_main_param[0][0][1] is None
