"""CPU tests for AdapterStore's register/delete state machine.

Pins the stale-live invariant: after the *current* adapter is deleted, the
live GPU state still mirrors the deleted tenant, so a subsequently created
adapter must be seeded from the pristine slot (and only become live via
swap_to) instead of silently inheriting the deleted tenant's weights.

The state-machine tests stub out tensor plumbing. The optimizer-state tests
exercise the snapshot and restore path with CPU tensors.
"""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


def _import_adapter_store_without_megatron_extensions():
    """Import the CPU-only state helpers without loading optional TE binaries."""
    target = "skyrl.backends.skyrl_train.workers.megatron.adapter_store"
    module_names = (
        target,
        "megatron",
        "megatron.core",
        "megatron.core.parallel_state",
        "megatron.core.distributed",
        "megatron.core.optimizer",
    )
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    megatron = ModuleType("megatron")
    megatron.__path__ = []
    core = ModuleType("megatron.core")
    core.__path__ = []
    parallel_state = ModuleType("megatron.core.parallel_state")
    distributed = ModuleType("megatron.core.distributed")
    optimizer = ModuleType("megatron.core.optimizer")
    distributed.DistributedDataParallel = type("DistributedDataParallel", (), {})
    optimizer.ChainedOptimizer = type("ChainedOptimizer", (), {})
    core.parallel_state = parallel_state
    try:
        sys.modules.pop(target, None)
        sys.modules["megatron"] = megatron
        sys.modules["megatron.core"] = core
        sys.modules["megatron.core.parallel_state"] = parallel_state
        sys.modules["megatron.core.distributed"] = distributed
        sys.modules["megatron.core.optimizer"] = optimizer
        return importlib.import_module(target)
    finally:
        for name, module in saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


adapter_store = _import_adapter_store_without_megatron_extensions()
AdapterStore = adapter_store.AdapterStore
_iter_optimizer_state_groups = adapter_store._iter_optimizer_state_groups

SIGNATURE = object()


def _store() -> tuple[AdapterStore, list[tuple[object, object]]]:
    """AdapterStore with tensor plumbing stubbed; returns (store, copy_log)."""
    store = AdapterStore()
    store._signature = SIGNATURE
    store._pristine = object()
    copies: list[tuple[object, object]] = []
    store._allocate_empty_slot = lambda model_chunks, optimizer: object()
    store._copy_slot = lambda src, dst: copies.append((src, dst))
    return store, copies


def test_first_create_treats_live_as_authoritative():
    store, copies = _store()

    store.create("model-a", model_chunks=[], optimizer=None, signature=SIGNATURE)

    assert store.current_id == "model-a"
    assert copies == []


def test_create_after_deleting_current_seeds_from_pristine():
    store, copies = _store()
    store.create("model-a", model_chunks=[], optimizer=None, signature=SIGNATURE)
    store.delete("model-a")

    store.create("model-b", model_chunks=[], optimizer=None, signature=SIGNATURE)

    # Live state still mirrors deleted model-a: model-b's slot must be a
    # pristine copy, and it must not be considered live until swap_to runs.
    assert copies == [(store._pristine, store._slots["model-b"])]
    assert store.current_id is None


def test_delete_of_non_current_adapter_keeps_live_authoritative():
    store, copies = _store()
    store.create("model-a", model_chunks=[], optimizer=None, signature=SIGNATURE)
    store.create("model-b", model_chunks=[], optimizer=None, signature=SIGNATURE)
    copies.clear()

    store.delete("model-b")
    store.create("model-c", model_chunks=[], optimizer=None, signature=SIGNATURE)

    # model-a is still current and live; model-c seeds from pristine as any
    # subsequent registration does.
    assert store.current_id == "model-a"
    assert copies == [(store._pristine, store._slots["model-c"])]


def _optimizer(precision_aware: bool = True, with_master: bool = True):
    optimizer_param = torch.ones(2, dtype=torch.bfloat16)
    main_param = torch.ones(2, dtype=torch.float32)
    state = {
        "exp_avg": torch.full((2,), 2.0),
        "exp_avg_sq": torch.full((2,), 3.0),
        "step": torch.tensor(4.0),
    }
    param_to_inner_param = {}
    if precision_aware:
        state_key = optimizer_param
        main_group = [None]
        if with_master:
            state["master_param"] = main_param
            param_to_inner_param[optimizer_param] = main_param
    else:
        state_key = main_param
        main_group = [main_param]
    inner = SimpleNamespace(
        param_groups=[{"params": [optimizer_param], "step": 4}],
        state={state_key: state},
        param_to_inner_param=param_to_inner_param,
    )
    return SimpleNamespace(
        shard_float16_groups=[[optimizer_param]],
        shard_fp32_from_float16_groups=[main_group],
        optimizer=inner,
    )


@pytest.mark.parametrize("precision_aware", [False, True])
def test_optimizer_state_round_trip(monkeypatch, precision_aware):
    monkeypatch.setattr(
        "skyrl.backends.skyrl_train.workers.megatron.adapter_store._new_pinned_like",
        lambda tensor: torch.empty_like(tensor, device="cpu"),
    )
    optimizer = _optimizer(precision_aware=precision_aware)
    store = AdapterStore()
    slot = store._allocate_empty_slot([], optimizer)
    store._snapshot(slot, [], optimizer)

    group = next(iter(_iter_optimizer_state_groups(optimizer)))
    main_param, state = group[0]
    assert "master_param" not in slot.cpu_opt_state[0][0][0]
    main_param.fill_(10.0)
    state["exp_avg"].fill_(20.0)
    state["exp_avg_sq"].fill_(30.0)
    state["step"].fill_(40.0)
    optimizer.optimizer.param_groups[0]["step"] = 40

    store._restore(slot, [], optimizer)

    assert torch.equal(main_param, torch.ones(2))
    assert torch.equal(state["exp_avg"], torch.full((2,), 2.0))
    assert torch.equal(state["exp_avg_sq"], torch.full((2,), 3.0))
    assert state["step"].item() == 4.0
    assert optimizer.optimizer.param_groups[0]["step"] == 4


def test_precision_aware_optimizer_requires_master_tensor():
    optimizer = _optimizer(with_master=False)

    with pytest.raises(RuntimeError, match="has no FP32 master tensor"):
        list(_iter_optimizer_state_groups(optimizer))
