"""LocalShardAdamW / local_shard_clip_grad_norm_ == torch AdamW / clip_grad_norm_ (2 gloo ranks, CPU)."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

from skyrl.backends.skyrl_train.distributed.fsdp_local_optim import (
    LocalShardAdamW,
    local_shard_clip_grad_norm_,
)

SHAPES = [(32, 48), (48, 32), (7, 16), (5,), (2, 3, 8)]  # uneven rows shard unevenly


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make(mesh, seed):
    g = torch.Generator().manual_seed(seed)
    full = [torch.randn(s, generator=g) for s in SHAPES]
    grads = [[torch.randn(s, generator=g) * 10 for s in SHAPES] for _ in range(3)]
    params = [torch.nn.Parameter(distribute_tensor(t.clone(), mesh, [Shard(0)])) for t in full]
    return full, grads, params


def _set_grads(params, grads, mesh):
    for p, gr in zip(params, grads):
        p.grad = distribute_tensor(gr.clone(), mesh, [Shard(0)])


def _worker(rank, world, port, kwargs, q):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        mesh = init_device_mesh("cpu", (world,))
        opt_kw = dict(lr=1e-2, betas=(0.9, 0.98), weight_decay=0.1, **kwargs)
        full, grads, ref_p = _make(mesh, 0)
        _, _, new_p = _make(mesh, 0)
        plain = [torch.nn.Parameter(t.clone()) for t in full]
        ref = torch.optim.AdamW(ref_p, **opt_kw)
        new = LocalShardAdamW(new_p, **opt_kw)
        plain_opt = torch.optim.AdamW(plain, **opt_kw)
        out = {"adam_eq_dtensor": True, "adam_eq_plain": True, "norm_eq": True, "clipped_eq": True}
        for it, gr in enumerate(grads):
            _set_grads(ref_p, gr, mesh)
            _set_grads(new_p, gr, mesh)
            for p, g in zip(plain, gr):
                p.grad = g.clone()
            n_ref = torch.nn.utils.clip_grad_norm_(ref_p, max_norm=1.0)
            n_new = local_shard_clip_grad_norm_(new_p, max_norm=1.0).cpu()  # on cuda when available
            n_plain = torch.nn.utils.clip_grad_norm_(plain, max_norm=1.0)
            n_ref = n_ref.full_tensor() if isinstance(n_ref, DTensor) else n_ref
            out["norm_eq"] &= torch.equal(n_ref, n_new) and torch.allclose(n_new, n_plain, rtol=1e-6)
            out["clipped_eq"] &= all(
                torch.equal(a.grad.full_tensor(), b.grad.full_tensor()) for a, b in zip(ref_p, new_p)
            )
            for p, b in zip(plain, new_p):  # same clipped grads (the norms differ in rounding only)
                p.grad = b.grad.full_tensor().clone()
            ref.step()
            new.step()
            plain_opt.step()
            out["adam_eq_dtensor"] &= all(torch.equal(a.full_tensor(), b.full_tensor()) for a, b in zip(ref_p, new_p))
            out["adam_eq_plain"] &= all(torch.equal(a.full_tensor(), b) for a, b in zip(new_p, plain))
        # state stays DTensor and round-trips through state_dict
        sd = new.state_dict()
        out["state_is_dtensor"] = all(isinstance(s["exp_avg"], DTensor) for s in sd["state"].values())
        new2 = LocalShardAdamW(new_p, **opt_kw)
        new2.load_state_dict(sd)
        out["state_roundtrip"] = all(
            torch.equal(new2.state[p]["exp_avg_sq"].full_tensor(), new.state[p]["exp_avg_sq"].full_tensor())
            for p in new_p
        )
        q.put((rank, out))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "kwargs",
    [dict(foreach=True), dict(foreach=False)],
    ids=["foreach", "single-tensor"],
)
def test_local_shard_adamw_and_clip_match_torch(kwargs):
    world = 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [ctx.Process(target=_worker, args=(r, world, port, kwargs, q)) for r in range(world)]
    for p in procs:
        p.start()
    results = dict(q.get(timeout=120) for _ in range(world))
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0
    for rank, out in results.items():
        assert all(out.values()), (rank, out)
