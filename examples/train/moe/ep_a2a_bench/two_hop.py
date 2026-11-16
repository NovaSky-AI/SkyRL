"""Expert-parallel (EP) dispatch and combine, standalone: ``DirectDispatch`` and ``TwoHopDispatch``.

Both take, on every EP rank, tokens ``x [T, H]``, top-k expert ids ``idx [T, k]`` and weights
``probs [T, k]``; hand the local experts their input rows grouped by expert; and return
``out[t] = sum_k probs[t, k] * expert_{idx[t, k]}(x[t])``.

DirectDispatch (what Megatron's alltoall dispatcher does): one row per (token, expert), sent
straight to the rank that owns the expert.

TwoHopDispatch (once per node):
  hop 1, network   the source sends each token ONCE per destination node, with its expert ids and
                   weights, to the relay: the rank with the same index on that node
                   (``node * ranks_per_node + rank % ranks_per_node``). For its own node the source
                   is the relay and nothing moves.
  hop 2, NVLink    the relay makes one row per (token, expert on this node) and sends it to the
                   owner rank.
  combine          owners return expert outputs to the relay (NVLink); the relay does the weighted
                   sum over its node's experts and returns ONE row per token (network); the source
                   adds the per-node partial sums.

Rows travel as raw bytes (uint8), so payload and metadata share one message per peer. The only
host syncs are the size reads that the transport needs. Assumes expert tensor parallel size 1 and
that EP rank ``r`` lives on node ``r // ranks_per_node``.
"""

import time
from contextlib import contextmanager
from typing import List

import torch
import torch.distributed as dist


class Comm:
    """Variable-size exchange inside a set of ranks. ``transport``: "a2a" (NCCL) or "p2p" (any backend)."""

    def __init__(self, ep: int, ranks_per_node: int, transport: str, device):
        assert ep % ranks_per_node == 0
        self.ep, self.rpn, self.transport, self.device = ep, ranks_per_node, transport, device
        self.rank = dist.get_rank()
        self.nodes = ep // ranks_per_node
        self.node, self.local = divmod(self.rank, ranks_per_node)
        self.all_ranks = list(range(ep))
        self.node_ranks = [self.node * ranks_per_node + i for i in range(ranks_per_node)]
        self.rail_ranks = [n * ranks_per_node + self.local for n in range(self.nodes)]
        self.groups = {tuple(self.all_ranks): None}
        if transport == "a2a":
            # Every process must create every group, in the same order.
            for n in range(self.nodes):
                ranks = [n * ranks_per_node + i for i in range(ranks_per_node)]
                self.groups[tuple(ranks)] = dist.new_group(ranks)
            for i in range(ranks_per_node):
                ranks = [n * ranks_per_node + i for n in range(self.nodes)]
                self.groups[tuple(ranks)] = dist.new_group(ranks)
        self.timers = {}
        self.sync = torch.cuda.synchronize if torch.device(device).type == "cuda" else (lambda: None)

    @contextmanager
    def phase(self, name: str):
        self.sync()
        t0 = time.perf_counter()
        yield
        self.sync()
        self.timers[name] = self.timers.get(name, 0.0) + time.perf_counter() - t0

    def exchange(self, peers: List[int], bufs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Send ``bufs[j]`` ([rows_j, W] uint8) to ``peers[j]``; return what each peer sent here."""
        n, me, width = len(peers), peers.index(self.rank), bufs[0].shape[1]
        if n == 1:
            return [bufs[0]]
        out_counts = [b.shape[0] for b in bufs]
        if self.transport == "a2a":
            group = self.groups[tuple(peers)]
            sizes_out = torch.tensor(out_counts, dtype=torch.int64, device=self.device)
            sizes_in = torch.empty_like(sizes_out)
            dist.all_to_all_single(sizes_in, sizes_out, group=group)
            in_counts = sizes_in.tolist()  # host sync: the transport needs the sizes
            recv = torch.empty(sum(in_counts), width, dtype=torch.uint8, device=self.device)
            dist.all_to_all_single(
                recv, torch.cat(bufs), output_split_sizes=in_counts, input_split_sizes=out_counts, group=group
            )
            return list(recv.split(in_counts))
        sizes_in = [torch.zeros(1, dtype=torch.int64, device=self.device) for _ in peers]
        ops = []
        for j, p in enumerate(peers):
            if j != me:
                ops.append(dist.P2POp(dist.irecv, sizes_in[j], p))
                ops.append(dist.P2POp(dist.isend, torch.tensor([out_counts[j]], device=self.device), p))
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        recv = [torch.empty(int(s), width, dtype=torch.uint8, device=self.device) for s in sizes_in]
        recv[me] = bufs[me]
        ops = []
        for j, p in enumerate(peers):
            if j != me and recv[j].shape[0]:
                ops.append(dist.P2POp(dist.irecv, recv[j], p))
            if j != me and bufs[j].shape[0]:
                ops.append(dist.P2POp(dist.isend, bufs[j].contiguous(), p))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return recv


def _bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8).reshape(t.shape[0], -1)


def _split(order_key: torch.Tensor, n: int):
    """Stable order that groups rows by ``order_key`` in [0, n), and the group sizes."""
    order = torch.argsort(order_key, stable=True)
    counts = torch.bincount(order_key, minlength=n).tolist()  # host sync: sizes for the transport
    return order, counts


class _Base:
    def __init__(self, comm: Comm, num_experts: int, hidden: int, topk: int, dtype):
        self.c, self.E, self.H, self.k, self.dtype = comm, num_experts, hidden, topk, dtype
        self.epr = num_experts // comm.ep  # experts per rank
        self.xb = hidden * torch.empty(0, dtype=dtype).element_size()  # bytes of one hidden vector

    def _x(self, raw: torch.Tensor) -> torch.Tensor:
        return raw[:, : self.xb].contiguous().view(self.dtype)

    def _ids(self, raw: torch.Tensor, off: int, n: int) -> torch.Tensor:
        return raw[:, off : off + 2 * n].contiguous().view(torch.int16).long()

    def _to_experts(self, raw: torch.Tensor):
        """Owner side: rows (x | expert id) in arrival order -> rows grouped by local expert."""
        ids = self._ids(raw, self.xb, 1).squeeze(1) - self.c.rank * self.epr
        self.perm = torch.argsort(ids, stable=True)
        return self._x(raw)[self.perm], torch.bincount(ids, minlength=self.epr)

    def _from_experts(self, y: torch.Tensor) -> List[torch.Tensor]:
        """Owner side: expert outputs back into arrival order, split per sender."""
        back = torch.empty_like(y)
        back[self.perm] = y
        return list(_bytes(back).split(self.owner_recv_counts))


class DirectDispatch(_Base):
    def dispatch(self, x, idx, probs):
        c = self.c
        with c.phase("direct/pack"):
            flat = idx.reshape(-1)
            self.order, counts = _split(flat // self.epr, c.ep)
            self.tok = torch.arange(x.shape[0], device=x.device).repeat_interleave(self.k)[self.order]
            self.w = probs.reshape(-1)[self.order]
            self.T = x.shape[0]
            rows = torch.cat([_bytes(x[self.tok]), _bytes(flat[self.order].to(torch.int16).unsqueeze(1))], dim=1)
        with c.phase("direct/transfer"):
            recv = c.exchange(c.all_ranks, list(rows.split(counts)))
        with c.phase("direct/sort"):
            self.owner_recv_counts = [r.shape[0] for r in recv]
            return self._to_experts(torch.cat(recv))

    def combine(self, y):
        c = self.c
        with c.phase("direct/unsort"):
            bufs = self._from_experts(y)
        with c.phase("direct/transfer_back"):
            recv = torch.cat(c.exchange(c.all_ranks, bufs))
        with c.phase("direct/sum"):
            out = torch.zeros(self.T, self.H, dtype=torch.float32, device=y.device)
            out.index_add_(0, self.tok, self._x(recv).float() * self.w.float().unsqueeze(1))
            return out.to(self.dtype)


class TwoHopDispatch(_Base):
    def dispatch(self, x, idx, probs):
        c = self.c
        with c.phase("twohop/pack"):
            node_of = idx // (self.epr * c.rpn)  # [T, k]
            self.T = x.shape[0]
            self.sel, bufs = [], []
            for n in range(c.nodes):
                sel = (node_of == n).any(dim=1).nonzero(as_tuple=True)[0]  # tokens with an expert on node n
                self.sel.append(sel)
                bufs.append(
                    torch.cat([_bytes(x[sel]), _bytes(idx[sel].to(torch.int16)), _bytes(probs[sel].float())], dim=1)
                )
        with c.phase("twohop/hop1_network"):
            recv = c.exchange(c.rail_ranks, bufs)
        with c.phase("twohop/expand"):
            self.relay_recv_counts = [r.shape[0] for r in recv]
            raw = torch.cat(recv)
            rx = self._x(raw)
            ridx = self._ids(raw, self.xb, self.k)
            rp = raw[:, self.xb + 2 * self.k :].contiguous().view(torch.float32)
            here = (ridx // (self.epr * c.rpn)) == c.node
            slot, _ = here.nonzero(as_tuple=True)  # relay-side token slot of every (token, expert) row
            e, w = ridx[here], rp[here]
            order, counts = _split(e // self.epr - c.node * c.rpn, c.rpn)
            self.relay_slot, self.relay_w, self.relay_T = slot[order], w[order], raw.shape[0]
            rows = torch.cat([_bytes(rx[self.relay_slot]), _bytes(e[order].to(torch.int16).unsqueeze(1))], dim=1)
        with c.phase("twohop/hop2_nvlink"):
            recv = c.exchange(c.node_ranks, list(rows.split(counts)))
        with c.phase("twohop/sort"):
            self.owner_recv_counts = [r.shape[0] for r in recv]
            return self._to_experts(torch.cat(recv))

    def combine(self, y):
        c = self.c
        with c.phase("twohop/unsort"):
            bufs = self._from_experts(y)
        with c.phase("twohop/hop2_nvlink_back"):
            recv = torch.cat(c.exchange(c.node_ranks, bufs))
        with c.phase("twohop/relay_sum"):
            part = torch.zeros(self.relay_T, self.H, dtype=torch.float32, device=y.device)
            part.index_add_(0, self.relay_slot, self._x(recv).float() * self.relay_w.unsqueeze(1))
            bufs = list(_bytes(part.to(self.dtype)).split(self.relay_recv_counts))
        with c.phase("twohop/hop1_network_back"):
            recv = c.exchange(c.rail_ranks, bufs)
        with c.phase("twohop/sum"):
            out = torch.zeros(self.T, self.H, dtype=torch.float32, device=y.device)
            for sel, r in zip(self.sel, recv):
                out.index_add_(0, sel, self._x(r).float())
            return out.to(self.dtype)
