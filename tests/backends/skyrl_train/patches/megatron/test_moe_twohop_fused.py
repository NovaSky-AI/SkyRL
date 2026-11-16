"""CPU check of the fused once-per-node MoE dispatch (``_moe_twohop_fused``) against a dense oracle.

Every rank runs in its own thread; the three collectives exchange tensors through a shared mailbox
and check each received split against the size the receiver computed from the count all-gather.
The kernels run in their plain-tensor reference form. A stand-in expert scales each row by its
probability and a per-expert constant, so the oracle is ``x * sum_e(routing * probs * scale)``.

uv run --isolated --extra skyrl-train --extra dev pytest \
    tests/backends/skyrl_train/patches/megatron/test_moe_twohop_fused.py
"""

import threading

import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron import _moe_twohop_fused as fused


class _Hub:
    def __init__(self, world: int):
        self.barrier = threading.Barrier(world, timeout=60)
        self.box = {}


class _ThreadComm(fused.Comm):
    def __init__(self, hub: _Hub, rank: int, num_nodes: int, node_size: int):
        self.hub, self.rank, self.num_nodes, self.node_size = hub, rank, num_nodes, node_size
        self.world_size = num_nodes * node_size
        node, local = divmod(rank, node_size)
        self.rail = [n * node_size + local for n in range(num_nodes)]
        self.node = [node * node_size + g for g in range(node_size)]

    def all_gather(self, vec):
        self.hub.box[("ag", self.rank)] = vec
        self.hub.barrier.wait()
        out = torch.stack([self.hub.box[("ag", r)] for r in range(self.world_size)])
        self.hub.barrier.wait()
        return out

    def _a2a(self, members, x, send, recv):
        assert sum(send) == x.size(0), (sum(send), x.shape)
        for dst, piece in zip(members, x.split(send)):
            self.hub.box[(self.rank, dst)] = piece
        self.hub.barrier.wait()
        pieces = [self.hub.box[(src, self.rank)] for src in members]
        assert [p.size(0) for p in pieces] == recv, ([p.size(0) for p in pieces], recv)
        out = torch.cat(pieces)
        self.hub.barrier.wait()
        return out

    def rail_a2a(self, x, send, recv):
        return self._a2a(self.rail, x, send, recv)

    def node_a2a(self, x, send, recv):
        return self._a2a(self.node, x, send, recv)


def _run(num_nodes, node_size, num_local, topk, x_dtype, p_dtype, hidden=24):
    world = num_nodes * node_size
    experts = world * num_local
    hub = _Hub(world)
    scale = torch.arange(1, experts + 1, dtype=torch.float32) / experts
    weight = torch.linspace(-1, 1, hidden)
    results, errors = {}, []

    def rank_main(rank):
        try:
            gen = torch.Generator().manual_seed(1000 + rank)
            tokens = 0 if rank == world - 1 and world > 2 else 13 + 7 * rank  # one rank without tokens
            x0 = torch.randn(tokens, hidden, generator=gen).to(x_dtype)
            top = torch.rand(tokens, experts, generator=gen).topk(topk, dim=-1).indices
            routing = torch.zeros(tokens, experts, dtype=torch.bool).scatter_(1, top, True)
            p0 = (torch.rand(tokens, experts, generator=gen) * routing).to(p_dtype)

            x, p = x0.clone().requires_grad_(True), p0.clone().requires_grad_(True)
            state = fused.TwoHopState(_ThreadComm(hub, rank, num_nodes, node_size), num_local, topk)
            h, pr = fused.dispatch(state, x, p, routing)
            local = torch.arange(rank * num_local, (rank + 1) * num_local)
            row_scale = torch.repeat_interleave(scale[local], state.tokens_per_expert)
            y = (h.float() * (pr.float() * row_scale).unsqueeze(-1)).to(x_dtype)
            out = fused.combine(state, y)
            (out.float() * weight).sum().backward()

            xr, pr_ref = x0.clone().requires_grad_(True), p0.clone().requires_grad_(True)
            ref = xr.float() * (pr_ref.float() * routing * scale).sum(dim=1, keepdim=True)
            (ref * weight).sum().backward()
            results[rank] = (out.detach().float(), ref.detach(), x.grad.float(), xr.grad.float(), p.grad, pr_ref.grad)
        except BaseException as e:  # noqa: BLE001 - release the other ranks, then report
            errors.append((rank, e))
            hub.barrier.abort()

    threads = [threading.Thread(target=rank_main, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    real = [(r, e) for r, e in errors if not isinstance(e, threading.BrokenBarrierError)]
    assert not real, real
    assert not errors, errors
    return results


@pytest.mark.parametrize(
    "num_nodes,node_size,num_local,topk",
    [(2, 2, 2, 3), (3, 2, 1, 2), (1, 4, 2, 4), (2, 1, 3, 2), (2, 4, 4, 6)],
)
@pytest.mark.parametrize(
    "x_dtype,p_dtype,tol",
    [(torch.float32, torch.float32, 1e-5), (torch.float32, torch.float64, 1e-5), (torch.bfloat16, torch.float32, 3e-2)],
)
def test_fused_twohop_matches_dense_oracle(num_nodes, node_size, num_local, topk, x_dtype, p_dtype, tol):
    for rank, (out, ref, dx, dx_ref, dp, dp_ref) in _run(
        num_nodes, node_size, num_local, topk, x_dtype, p_dtype
    ).items():
        for name, a, b in (("out", out, ref), ("dx", dx, dx_ref), ("dprobs", dp.double(), dp_ref.double())):
            assert a.shape == b.shape, (rank, name, a.shape, b.shape)
            if b.numel():
                err = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
                assert err < tol, (rank, name, err)
