"""Unit tests for AdapterStore swap semantics around offload and delete.

Regressions covered (both hit in production on 2026-08-18):

1. swap_to() while the DDP grad buffers are offloaded — Megatron frees
   grad_data via storage().resize_(0) after every post-step offload, and the
   H2D copy_ into the freed view failed with CUDA `invalid argument`.
2. create() after delete-of-current — the store adopted the live GPU state
   (the *deleted* tenant's weights, fp32 masters and Adam state) as the new
   adapter instead of seeding from pristine.

The tests fake the Megatron DDP buffers by monkeypatching _iter_buffers, so
they run on a single GPU with KB-scale allocations and no distributed init.
"""

import pytest
import torch
from types import SimpleNamespace

from skyrl.backends.skyrl_train.workers.megatron import adapter_store as astore
from skyrl.backends.skyrl_train.workers.megatron.adapter_store import (
    AdapterStore,
    LoraSignature,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_SIG = LoraSignature(
    rank=8,
    alpha=16,
    target_modules=("linear_proj",),
    lora_type="lora",
    tp_size=1,
    pp_size=1,
    ep_size=1,
)


class _FakeBuf:
    """Stands in for a megatron _ParamAndGradBuffer (param_data + grad_data)."""

    def __init__(self, n: int = 64):
        self.param_data = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        self.grad_data = torch.zeros(n, device="cuda", dtype=torch.bfloat16)
        self.params = []  # _expected_lora_param_check iterates this

    def free_grad_storage(self):
        """Mimic DDP.offload_grad_buffers(): free storage, keep the view."""
        torch.cuda.synchronize()
        self.grad_data.untyped_storage().resize_(0)


def _fake_opt():
    return SimpleNamespace(optimizer=SimpleNamespace(state={}, param_groups=[]))


@pytest.fixture
def store_env(monkeypatch):
    """AdapterStore wired to one fake buffer; model_chunks is [buf] itself."""
    buf = _FakeBuf()
    monkeypatch.setattr(astore, "_iter_buffers", lambda chunks: [(0, 0, chunks[0])])
    monkeypatch.setattr(astore, "_expected_lora_param_check", lambda chunks: None)
    store = AdapterStore()
    opt = _fake_opt()
    store.register_pristine([buf], opt, _SIG)
    return store, buf, opt


def test_swap_with_offloaded_grads_does_not_crash(store_env):
    """Prod crash: snapshot/restore must skip grad copies on freed storage."""
    store, buf, opt = store_env
    store.create("A", [buf], opt, _SIG)  # first adapter: adopts live
    store.create("B", [buf], opt, _SIG)  # seeded from pristine
    pristine_params = buf.param_data.clone()

    # "Train" A, then offload grads exactly like offload_after_step does.
    buf.param_data.add_(1.0)
    buf.free_grad_storage()

    # Previously: torch.AcceleratorError CUDA invalid argument in _restore.
    store.swap_to("B", [buf], opt)

    assert store.current_id == "B"
    torch.cuda.synchronize()
    # B was seeded from pristine, so live params must be back to pristine.
    assert torch.equal(buf.param_data, pristine_params)
    # A's slot still captured the trained params (param storage was live) ...
    a_params = store._slots["A"].cpu_param_data[0][0]
    assert torch.equal(a_params, (pristine_params + 1.0).cpu())
    # ... while its grads were recorded as zero instead of reading freed memory.
    a_grads = store._slots["A"].cpu_grad_data[0][0]
    assert torch.count_nonzero(a_grads) == 0


def test_create_after_delete_does_not_adopt_live_state(store_env):
    """Prod bug: a new adapter created after delete-of-current inherited the
    deleted tenant's live state instead of pristine."""
    store, buf, opt = store_env
    pristine_params = buf.param_data.clone()

    store.create("A", [buf], opt, _SIG)  # adopts live (true first create)
    buf.param_data.add_(2.0)  # "train" A
    store.delete("A")  # session expiry: current cleared, live now stale

    store.create("B", [buf], opt, _SIG)
    # B must NOT adopt the deleted tenant's live state.
    assert store.current_id is None

    store.swap_to("B", [buf], opt)
    torch.cuda.synchronize()
    assert store.current_id == "B"
    assert torch.equal(buf.param_data, pristine_params)


def test_full_prod_sequence_create_before_expiry_delete(store_env):
    """The exact 06:57 ordering: create B while A is current, then A expires
    (delete -> current=None), then swap_to(B) with grads offloaded."""
    store, buf, opt = store_env
    pristine_params = buf.param_data.clone()

    store.create("A", [buf], opt, _SIG)
    buf.param_data.add_(3.0)  # overnight training
    store.create("B", [buf], opt, _SIG)  # new session registers first...
    store.delete("A")  # ...then the stale session expires
    buf.free_grad_storage()  # trainer idle: grads offloaded

    store.swap_to("B", [buf], opt)
    torch.cuda.synchronize()
    assert store.current_id == "B"
    assert torch.equal(buf.param_data, pristine_params)

    # A subsequent adapter created while B is live seeds from pristine and
    # round-trips through a snapshot of B without touching freed grads.
    store.create("C", [buf], opt, _SIG)
    store.swap_to("C", [buf], opt)
    torch.cuda.synchronize()
    assert store.current_id == "C"
    assert torch.equal(buf.param_data, pristine_params)
