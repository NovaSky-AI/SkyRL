"""Exact-size host offload (``distributed/host_offload.py``) on CPU.

The production path offloads CUDA storages into a ``cudaHostRegister``-ed buffer; here the same
code runs with ``device_type="cpu"`` (no pinning), which exercises everything but the copy
engine: storage selection and dedup, resize to 0 in place, byte layout, restore, release.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl.backends.skyrl_train.distributed import host_offload as ho


def _model():
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(13, 7), torch.nn.LayerNorm(7), torch.nn.Linear(7, 3))
    m.register_buffer("scale", torch.randn(5))
    return m


def _snapshot(module, opt=None):
    out = {f"p.{n}": p.detach().clone() for n, p in module.named_parameters()}
    out |= {f"g.{n}": p.grad.clone() for n, p in module.named_parameters() if p.grad is not None}
    out |= {f"b.{n}": b.clone() for n, b in module.named_buffers()}
    if opt is not None:
        for i, st in enumerate(opt.state.values()):
            out |= {f"o.{i}.{k}": v.clone() for k, v in st.items()}
    return out


def _step(m, opt, seed):
    g = torch.Generator().manual_seed(seed)
    loss = m(torch.randn(4, 13, generator=g)).square().mean()
    loss.backward()
    opt.step()


def _assert_bitwise(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype and torch.equal(
            a[k].reshape(-1).view(torch.uint8), b[k].reshape(-1).view(torch.uint8)
        ), k


class TestModuleAndOptimizerRoundTrip:
    def test_bitwise_and_accounting(self):
        m = _model()
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        _step(m, opt, 1)  # grads and Adam state (incl. the CPU step scalar) exist
        before = _snapshot(m, opt)
        w_view = m[0].weight[2:5]  # a view must see the restored bytes, not a stale copy

        storages = {
            t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in ho.module_tensors(m) if t is not None
        }
        held = ho.offload_to_host(m, ho.module_tensors(m), device_type="cpu")
        held_opt = ho.offload_to_host(opt, ho.optimizer_tensors(opt), device_type="cpu")
        state = ho._OFFLOADED[m]
        assert held == sum((n + ho._ALIGN - 1) // ho._ALIGN * ho._ALIGN for n in storages.values())
        assert held < sum(storages.values()) + ho._ALIGN * len(storages)
        assert held_opt > 0 and ho.offloaded_bytes(opt) == held_opt
        assert all(p.untyped_storage().nbytes() == 0 for p in m.parameters())
        assert ho.offload_to_host(m, ho.module_tensors(m), device_type="cpu") == 0  # already offloaded

        assert ho.onload_from_host(m) == held and ho.onload_from_host(opt) == held_opt
        assert state.buffer._mmap.closed and not ho.is_offloaded(m)
        assert ho.onload_from_host(m) == 0
        _assert_bitwise(before, _snapshot(m, opt))
        assert torch.equal(w_view, before["p.0.weight"][2:5])

    def test_training_after_round_trips_matches_no_offload(self):
        ref, m = _model(), _model()
        ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-2)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        for step in range(3):
            _step(ref, ref_opt, step)
            _step(m, opt, step)
            ho.offload_to_host(m, ho.module_tensors(m), device_type="cpu")
            ho.offload_to_host(opt, ho.optimizer_tensors(opt), device_type="cpu")
            ho.onload_from_host(opt)
            ho.onload_from_host(m)
            ref_opt.zero_grad(set_to_none=True)
            opt.zero_grad(set_to_none=True)
        _assert_bitwise(_snapshot(ref, ref_opt), _snapshot(m, opt))

    def test_storage_shared_by_views_offloaded_once(self):
        base, owner = torch.arange(10, dtype=torch.float32), torch.nn.Module()
        held = ho.offload_to_host(owner, [base, base[3:], base.view(2, 5)], device_type="cpu")
        assert held == 64  # 40 bytes, one storage, aligned once
        ho.onload_from_host(owner)
        assert torch.equal(base, torch.arange(10, dtype=torch.float32))


def _fsdp_worker(rank, world, out_dir):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.testing._internal.distributed.fake_pg import FakeStore

    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    m = _model()  # 13x7 / 7x3 weights: uneven rows over 2 ranks -> FSDP2 pads the local shard
    for layer in (m[0], m[2]):
        fully_shard(layer, mesh=mesh)
    fully_shard(m, mesh=mesh)
    before = _snapshot(m)
    fsdp_params = [fp for g in m._get_fsdp_state()._fsdp_param_groups for fp in g.fsdp_params]
    padded_before = [fp._sharded_param_data.clone() for fp in fsdp_params]
    ho.offload_module(m, device_type="cpu")
    assert all(fp._sharded_param_data.untyped_storage().nbytes() == 0 for fp in fsdp_params)
    ho.onload_module(m)
    after = _snapshot(m)
    ok = before.keys() == after.keys() and all(torch.equal(before[k], after[k]) for k in before)
    ok &= all(torch.equal(a, fp._sharded_param_data) for a, fp in zip(padded_before, fsdp_params))
    # The DTensor's local tensor still aliases FSDP2's padded buffer (no re-registration needed).
    ok &= all(
        fp.sharded_param._local_tensor.untyped_storage().data_ptr()
        == fp._sharded_param_data.untyped_storage().data_ptr()
        for fp in fsdp_params
    )
    torch.save(ok, os.path.join(out_dir, f"ok{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="no torch.distributed")
def test_fsdp2_sharded_round_trip(tmp_path):
    mp.spawn(_fsdp_worker, args=(2, str(tmp_path)), nprocs=2)
    assert all(torch.load(tmp_path / f"ok{r}.pt") for r in range(2))


class TestFrozenOnlyOffload:
    """LoRA in-memory sync reads the adapters while the frozen base is offloaded: ``frozen_only``
    must free every frozen storage, keep the trainable ones (and their grads) readable in place,
    and restore the frozen ones bitwise."""

    def test_trainable_stay_frozen_restored(self):
        m = _model()
        for p in m[0].parameters():
            p.requires_grad_(False)
        m(torch.randn(2, 13)).sum().backward()
        before = _snapshot(m)
        trainable = {n: p for n, p in m.named_parameters() if p.requires_grad}
        frozen = {n: p for n, p in m.named_parameters() if not p.requires_grad}

        n = ho.offload_module(m, device_type="cpu", pin=False, frozen_only=True)

        assert n > 0
        assert all(p.untyped_storage().nbytes() == 0 for p in frozen.values())
        assert all(b.untyped_storage().nbytes() == 0 for b in m.buffers())
        for name, p in trainable.items():
            assert torch.equal(p.detach(), before[f"p.{name}"]), name
            assert torch.equal(p.grad, before[f"g.{name}"]), name
        ho.onload_module(m)
        _assert_bitwise(_snapshot(m), before)
