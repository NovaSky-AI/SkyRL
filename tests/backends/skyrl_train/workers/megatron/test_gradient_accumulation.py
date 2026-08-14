from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("megatron")

import skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper as wrapper_module
from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    MegatronPolicyWorkerBase,
)


def test_grad_sync_is_deferred_and_combines_token_counts(monkeypatch):
    calls = []
    monkeypatch.setattr(
        wrapper_module,
        "finalize_model_grads",
        lambda model, tokens: calls.append((model, tokens)),
    )

    wrapper = wrapper_module.MegatronModelWrapper.__new__(wrapper_module.MegatronModelWrapper)
    wrapper.actor_module = [object()]
    wrapper._pending_grad_sync = None

    wrapper._defer_finalize_model_grads(wrapper.actor_module, torch.tensor(3))
    wrapper._defer_finalize_model_grads(wrapper.actor_module, torch.tensor(5))
    assert calls == []

    wrapper.run_pending_grad_sync()

    assert calls[0][0] is wrapper.actor_module
    assert calls[0][1].item() == 8
    assert wrapper._pending_grad_sync is None


def test_optimizer_syncs_then_steps_then_clears_ddp_grad_buffer():
    events = []
    chunk = SimpleNamespace(zero_grad_buffer=lambda: events.append("clear"))
    model = SimpleNamespace(run_pending_grad_sync=lambda: events.append("sync"))
    strategy = SimpleNamespace(
        optimizer_step=lambda optimizer, model, scheduler, name: (events.append("step") or torch.tensor(2.5))
    )

    worker = MegatronPolicyWorkerBase.__new__(MegatronPolicyWorkerBase)
    worker.model = model
    worker.actor_module = [chunk]
    worker.strategy = strategy
    worker.optimizer = object()
    worker.scheduler = object()
    worker._micro_batches_accumulated = 3

    assert worker.optim_step() == pytest.approx(2.5)
    assert events == ["sync", "step", "clear"]
    assert worker._micro_batches_accumulated == 0
