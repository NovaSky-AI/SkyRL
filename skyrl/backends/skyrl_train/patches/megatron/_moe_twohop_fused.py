"""Fused once-per-node (two-hop) MoE dispatch and combine, independent of megatron-core.

Terms. A *source* rank holds tokens. The *relay* of a source on a node is the rank on that node
with the same local index. An *owner* rank holds experts. The *rail* group has one rank per node
(same local index) and crosses the network. The *node* group has the ranks of one node (NVLink).

Dispatch (2 all-to-alls, 1 count all-gather):

1. plan: every rank all-gathers its tokens-per-expert and its rows-per-destination-node. These
   numbers give every split size of both hops on every rank, so no other count exchange exists.
2. pack (source): one row per (token, destination node): the hidden state, the token's top-k
   probabilities and, per top-k slot, the expert column on that node (-1 if on another node).
3. hop 1: rail all-to-all.
4. expand (relay): one row per (token, expert) on this node: hidden state and one probability,
   grouped by owner rank and ordered by local expert.
5. hop 2: node all-to-all.
6. place (owner): splits rows into hidden states and probabilities, ordered by local expert.

Combine (2 all-to-alls): un-place (owner), node all-to-all, relay sum (one row per (token, node)),
rail all-to-all, final sum (source). The sums are gathers with a fixed fan-in (top-k at the relay,
number of nodes at the source), see ``_moe_twohop_kernels``.

The backward of each direction is written by hand and mirrors the other direction: the backward of
dispatch is a combine that also carries the probability gradients, and the backward of combine is a
dispatch without probabilities.

Messages that mix dtypes (hidden states, probabilities, int16 columns) travel as one ``uint8``
buffer; each field starts at an offset aligned to its dtype, and the row size is a multiple of 8.
"""

import contextlib
from typing import List

import numpy as np
import torch

from skyrl.backends.skyrl_train.patches.megatron._moe_twohop_kernels import (
    gather_rows,
    gather_sum_rows,
)


class Comm:
    """Transport of one rank. ``rank`` is the index in the all-gather order, which is
    ``node * node_size + local``. Subclasses provide the three collectives."""

    world_size: int
    rank: int
    num_nodes: int
    node_size: int

    def all_gather(self, vec: torch.Tensor) -> torch.Tensor:
        """[m] int64 -> [world_size, m], rows in rank order."""
        raise NotImplementedError

    def rail_a2a(self, x: torch.Tensor, send: List[int], recv: List[int]) -> torch.Tensor:
        """All-to-all over the rail group; splits are rows per node, in node order."""
        raise NotImplementedError

    def node_a2a(self, x: torch.Tensor, send: List[int], recv: List[int]) -> torch.Tensor:
        """All-to-all over the node group; splits are rows per local index."""
        raise NotImplementedError

    def phase(self, name: str):
        return contextlib.nullcontext()


def _round_up(n: int, m: int) -> int:
    return -(-n // m) * m


class _Layout:
    """Byte layout of the packed rows. Hop-1 row: hidden | K probabilities | K int16 columns.
    Hop-2 row: hidden | 1 probability."""

    def __init__(self, hidden: int, x_dtype: torch.dtype, p_dtype: torch.dtype, topk: int):
        xs, ps = x_dtype.itemsize, p_dtype.itemsize
        if xs not in (2, 4) or ps not in (2, 4, 8):
            raise NotImplementedError(f"fused two-hop dispatch: unsupported dtypes {x_dtype}, {p_dtype}")
        self.x_dtype, self.p_dtype, self.topk = x_dtype, p_dtype, topk
        self.hidden_words = hidden * xs // 2
        self.p_slot = _round_up(hidden * xs, ps) // ps  # first probability, in p_dtype elements
        self.col_slot = (self.p_slot + topk) * ps // 2  # first column, in int16 elements
        self.row1 = _round_up(self.col_slot * 2 + topk * 2, 8)
        self.row2 = _round_up((self.p_slot + 1) * ps, 8)

    @staticmethod
    def words(buf: torch.Tensor) -> torch.Tensor:
        return buf.view(torch.int16)

    def typed(self, buf: torch.Tensor) -> torch.Tensor:
        return buf.view(self.x_dtype)

    def probs1(self, buf: torch.Tensor) -> torch.Tensor:
        return buf.view(self.p_dtype)[:, self.p_slot : self.p_slot + self.topk]

    def prob2(self, buf: torch.Tensor) -> torch.Tensor:
        return buf.view(self.p_dtype)[:, self.p_slot]

    def cols(self, buf: torch.Tensor) -> torch.Tensor:
        return buf.view(torch.int16)[:, self.col_slot : self.col_slot + self.topk]


def _rows(src: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``src[idx]`` for a 2-D float tensor, through the word kernel."""
    src = src.contiguous()
    out = src.new_empty(idx.numel(), src.size(1))
    gather_rows(src.view(torch.int16), idx, out.view(torch.int16), out.view(torch.int16).size(1))
    return out


class TwoHopState:
    """One dispatch and its combine on one rank: the index maps built in ``dispatch_forward`` serve
    the combine and both backward passes."""

    def __init__(self, comm: Comm, num_local_experts: int, topk: int):
        self.comm, self.num_local, self.topk = comm, num_local_experts, topk

    # ------------------------------------------------------------------ dispatch
    def dispatch_forward(self, x: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor):
        c, K, L = self.comm, self.topk, self.num_local
        N, G = c.num_nodes, c.node_size
        W, E = G * L, N * G * L
        n0, l0 = divmod(c.rank, G)
        T, H = x.shape
        dev = x.device
        if routing_map.shape != (T, E):
            raise RuntimeError(f"fused two-hop dispatch: routing map {tuple(routing_map.shape)}, expected {(T, E)}")
        if W > 32767:
            raise NotImplementedError("fused two-hop dispatch: more than 32767 experts on one node")
        x = x.detach().contiguous()
        probs = probs.detach()
        lay = self.lay = _Layout(H, x.dtype, probs.dtype, K)
        self.T, self.H, self.E = T, H, E

        with c.phase("fwd/0_plan_counts_and_peer_wait"):
            eidx = routing_map.nonzero(as_tuple=True)[1]  # ordered by token, then expert
            if eidx.numel() != T * K:
                raise RuntimeError(
                    f"fused two-hop dispatch: {eidx.numel()} (token, expert) pairs for {T} tokens, expected top-{K}"
                )
            eidx = eidx.view(T, K)
            node_slot = torch.div(eidx, W, rounding_mode="floor")  # [T, K] destination node per slot
            touch = torch.zeros(T, N, dtype=torch.bool, device=dev).scatter_(1, node_slot, True)
            counts = c.all_gather(torch.cat([routing_map.sum(dim=0), touch.sum(dim=0)]).long()).cpu().numpy()
            cnt, c1 = counts[:, :E], counts[:, E:]  # [source, expert], [source, destination node]
            rail = np.arange(N) * G + l0  # the sources this rank relays for, in node order
            self.send1, self.recv1 = c1[c.rank].tolist(), c1[rail, n0].tolist()
            R1, P = sum(self.send1), sum(self.recv1)
            # Relay: copies per expert column of this node, from the sources on this rail.
            cc = cnt[rail, n0 * W : (n0 + 1) * W].sum(axis=0)
            self.send2, C = cc.reshape(G, L).sum(axis=1).tolist(), int(cc.sum())
            # Owner: rows per (relay local index, local expert).
            chunk = cnt[:, c.rank * L : (c.rank + 1) * L].reshape(N, G, L).sum(axis=0)
            self.recv2, D = chunk.sum(axis=1).tolist(), int(chunk.sum())
            self.tokens_per_expert = torch.from_numpy(chunk.sum(axis=0).astype(np.int64))
            # Rows arrive ordered (relay, expert); the experts need (expert, relay).
            in_start = (np.cumsum(chunk.reshape(-1)) - chunk.reshape(-1)).reshape(G, L)
            len_out = chunk.T.reshape(-1)
            out_end = np.cumsum(len_out)
            shift = in_start.T.reshape(-1) - (out_end - len_out)
            table = torch.as_tensor(np.stack([out_end, shift]), device=dev)
            q = torch.arange(D, device=dev)
            self.perm = q + table[1][torch.searchsorted(table[0], q, right=True).clamp_max(G * L - 1)]
            self.inv_perm = torch.empty_like(q).scatter_(0, self.perm, q)
            self.R1, self.P, self.C, self.D = R1, P, C, D

        with c.phase("fwd/1_pack"):
            node1, tok1 = touch.T.nonzero(as_tuple=True)  # hop-1 rows, ordered by node, then token
            if tok1.numel() != R1:
                raise RuntimeError(f"fused two-hop dispatch: {tok1.numel()} hop-1 rows, counted {R1}")
            pos1 = torch.full((T, N), -1, dtype=torch.long, device=dev)
            pos1[tok1, node1] = torch.arange(R1, device=dev)
            self.eidx, self.node_slot, self.tok1, self.pos1 = eidx, node_slot, tok1, pos1
            send = torch.empty(R1, lay.row1, dtype=torch.uint8, device=dev)
            gather_rows(x.view(torch.int16), tok1, lay.words(send), lay.hidden_words)
            lay.probs1(send).copy_(probs.gather(1, eidx).index_select(0, tok1))
            col = (eidx - node_slot * W).index_select(0, tok1)
            on_node = node_slot.index_select(0, tok1) == node1.unsqueeze(1)
            lay.cols(send).copy_(torch.where(on_node, col, -1).to(torch.int16))
        with c.phase("fwd/2_hop1_network"):
            pool = c.rail_a2a(send, self.send1, self.recv1)
        with c.phase("fwd/3_expand"):
            # Pool rows are ordered (source node, token). A stable sort of the slots by column gives
            # the hop-2 order: owner rank, local expert, source node, token.
            colr = lay.cols(pool)
            order = torch.sort(torch.where(colr >= 0, colr, W).reshape(-1), stable=True).indices
            src2 = order[:C]  # flat (pool row, slot) of each copy
            self.row2 = torch.div(src2, K, rounding_mode="floor")
            pos2 = torch.full((P * K,), -1, dtype=torch.long, device=dev)
            pos2[src2] = torch.arange(C, device=dev)
            self.pos2 = pos2.view(P, K)
            send = torch.empty(C, lay.row2, dtype=torch.uint8, device=dev)
            gather_rows(lay.words(pool), self.row2, lay.words(send), lay.hidden_words)
            lay.prob2(send).copy_(lay.probs1(pool).reshape(-1).index_select(0, src2))
        with c.phase("fwd/4_hop2_nvlink"):
            got = c.node_a2a(send, self.send2, self.recv2)
        with c.phase("fwd/5_place"):
            x_out = x.new_empty(D, H)
            gather_rows(lay.words(got), self.perm, x_out.view(torch.int16), lay.hidden_words)
            p_out = lay.prob2(got).index_select(0, self.perm)
        return x_out, p_out

    def dispatch_backward(self, gx: torch.Tensor, gp: torch.Tensor):
        c, lay, K = self.comm, self.lay, self.topk
        dev = gx.device
        gx = gx.to(lay.x_dtype).contiguous()
        with c.phase("bwd/dispatch/a_unplace"):
            send = torch.empty(self.D, lay.row2, dtype=torch.uint8, device=dev)
            gather_rows(gx.view(torch.int16), self.inv_perm, lay.words(send), lay.hidden_words)
            lay.prob2(send).copy_(gp.to(lay.p_dtype).index_select(0, self.inv_perm))
        with c.phase("bwd/dispatch/b_hop2_nvlink"):
            got = c.node_a2a(send, self.recv2, self.send2)
        with c.phase("bwd/dispatch/c_relay_sum"):
            send = torch.empty(self.P, lay.row1, dtype=torch.uint8, device=dev)
            gather_sum_rows(lay.typed(got), self.pos2, lay.typed(send), self.H)
            if self.C:
                g = lay.prob2(got).index_select(0, self.pos2.reshape(-1).clamp_min(0)).view(self.P, K)
                lay.probs1(send).copy_(g * (self.pos2 >= 0))
            else:
                lay.probs1(send).zero_()
        with c.phase("bwd/dispatch/d_hop1_network"):
            back = c.rail_a2a(send, self.recv1, self.send1)
        with c.phase("bwd/dispatch/e_final_sum"):
            gx_in = gx.new_empty(self.T, self.H)
            gather_sum_rows(lay.typed(back), self.pos1, gx_in, self.H)
            # Slot k of a token went to the hop-1 row of its node; its gradient returns in slot k.
            row = self.pos1.gather(1, self.node_slot)
            g = lay.probs1(back)[row, torch.arange(K, device=dev)] if self.T else gp.new_empty(0, K)
            gp_in = torch.zeros(self.T, self.E, dtype=lay.p_dtype, device=dev).scatter_(1, self.eidx, g)
        return gx_in, gp_in

    # ------------------------------------------------------------------ combine
    def combine_forward(self, y: torch.Tensor) -> torch.Tensor:
        c = self.comm
        y = y.detach()
        with c.phase("fwd/a_unplace"):
            send = _rows(y, self.inv_perm)
        with c.phase("fwd/b_back_hop2_nvlink"):
            got = c.node_a2a(send, self.recv2, self.send2)
        with c.phase("fwd/c_relay_sum"):
            partial = y.new_empty(self.P, y.size(1))
            gather_sum_rows(got, self.pos2, partial, y.size(1))
        with c.phase("fwd/d_back_hop1_network"):
            back = c.rail_a2a(partial, self.recv1, self.send1)
        with c.phase("fwd/e_final_sum"):
            out = y.new_empty(self.T, y.size(1))
            gather_sum_rows(back, self.pos1, out, y.size(1))
        return out

    def combine_backward(self, g: torch.Tensor) -> torch.Tensor:
        c = self.comm
        with c.phase("bwd/combine/1_pack"):
            send = _rows(g, self.tok1)
        with c.phase("bwd/combine/2_hop1_network"):
            pool = c.rail_a2a(send, self.send1, self.recv1)
        with c.phase("bwd/combine/3_expand"):
            send = _rows(pool, self.row2)
        with c.phase("bwd/combine/4_hop2_nvlink"):
            got = c.node_a2a(send, self.send2, self.recv2)
        with c.phase("bwd/combine/5_place"):
            return _rows(got, self.perm)


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, probs, routing_map, state: TwoHopState):
        ctx.state = state
        return state.dispatch_forward(x, probs, routing_map)

    @staticmethod
    def backward(ctx, gx, gp):
        gx_in, gp_in = ctx.state.dispatch_backward(gx, gp)
        return gx_in, gp_in, None, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, state: TwoHopState):
        ctx.state = state
        return state.combine_forward(y)

    @staticmethod
    def backward(ctx, g):
        return ctx.state.combine_backward(g.contiguous()), None


def dispatch(state: TwoHopState, x: torch.Tensor, probs: torch.Tensor, routing_map: torch.Tensor):
    """[T, H] hidden states and [T, E] probabilities -> rows of this rank's experts, ordered by
    local expert, and one probability per row. ``state.tokens_per_expert`` holds the row counts."""
    return _Dispatch.apply(x, probs, routing_map, state)


def combine(state: TwoHopState, y: torch.Tensor) -> torch.Tensor:
    """Expert outputs in dispatch order -> [T, H], the sum of each token's expert outputs."""
    return _Combine.apply(y, state)
