"""Node-deduplicated MoE all-to-all for ``MoEAlltoAllTokenDispatcher`` (opt-in).

megatron-core's all-to-all dispatcher sends one copy of a token per selected expert. With top-8
routing over experts spread across nodes, a token usually picks several experts on the same remote
node and crosses the inter-node network once per copy. When the inter-node network is the
bottleneck (e.g. NCCL over TCP sockets), that duplication dominates the step.

``SKYRL_MOE_NODE_DEDUP=1`` replaces the EP all-to-all with two hops:

1. **inter-node (rail) hop:** each token is sent once to every *node* hosting one of its experts,
   to the GPU there with the same local index (one group per local index, one rank per node),
   together with its routing bits and probabilities for that node's experts;
2. **intra-node hop:** that relay GPU fans the token out, one copy per expert, to the expert
   ranks on its node over NVLink.

Expert ranks end up with exactly the rows, order and probabilities megatron-core's dispatcher
gives them, so the ETP all-gather, the per-expert sort, the experts and ``combine_preprocess`` are
unchanged. The combine runs in reverse: expert outputs go back to the relay, which sums each
token's copies for its node, and one partial sum per (token, node) crosses back, where the
partials are summed. megatron-core's combine is the same unweighted bf16 sum over a token's copies
(probabilities are applied inside the experts), so only the order of the additions changes. All
hops are autograd ops, so the backward pass is deduplicated the same way.

Requirements (checked): dropless routing, unfused permutation, no quantization padding, no
batch-invariant mode, no shared-expert overlap, expert-data-parallel size 1 (EP * ETP == world),
ETP ranks contiguous, and nodes of equal size divisible by ETP. Node membership comes from the
hostname; ``SKYRL_MOE_NODE_SIZE`` overrides it (for single-node tests with pseudo-nodes).
"""

import functools
import os
import socket
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from loguru import logger

_ENV = "SKYRL_MOE_NODE_DEDUP"
_APPLIED = False
_TOPOLOGY: Dict[tuple, "_Topology"] = {}


class _Topology:
    """Rail (one rank per node, same local index) and node groups covering the expert ranks."""

    def __init__(self, ep_ranks: List[int], etp_size: int):
        world = dist.get_world_size()
        rank = dist.get_rank()
        node_size_env = os.environ.get("SKYRL_MOE_NODE_SIZE")
        if node_size_env:
            node_size = int(node_size_env)
            node_of = [r // node_size for r in range(world)]
        else:
            hosts: List[Optional[str]] = [None] * world
            dist.all_gather_object(hosts, socket.gethostname())
            order = {h: i for i, h in enumerate(dict.fromkeys(hosts))}
            node_of = [order[h] for h in hosts]
        num_nodes = max(node_of) + 1
        members = [[r for r in range(world) if node_of[r] == n] for n in range(num_nodes)]
        sizes = {len(m) for m in members}
        if len(sizes) != 1:
            raise RuntimeError(f"{_ENV}: nodes have different GPU counts: {[len(m) for m in members]}")
        node_size = sizes.pop()
        for m in members:
            if m != list(range(m[0], m[0] + node_size)):
                raise RuntimeError(f"{_ENV}: ranks of a node must be contiguous, got {m}")
        if node_size % etp_size:
            raise RuntimeError(f"{_ENV}: node size {node_size} not divisible by ETP {etp_size}")
        if len(ep_ranks) * etp_size != world:
            raise RuntimeError(f"{_ENV}: needs expert-data-parallel size 1 (EP * ETP == world size)")

        self.node_size, self.num_nodes = node_size, num_nodes
        self.node_of = node_of
        self.node = node_of[rank]
        self.local = rank - members[self.node][0]
        # new_group is collective: every rank creates every group in the same order.
        self.rail_group = None
        for local in range(node_size):
            ranks = [members[n][local] for n in range(num_nodes)]
            g = dist.new_group(ranks)
            if local == self.local:
                self.rail_group = g
        self.node_group = None
        for n in range(num_nodes):
            g = dist.new_group(members[n])
            if n == self.node:
                self.node_group = g
        # Expert ranks of *this* EP group per node, and each EP rank's (node, local index).
        self.ep_ranks = ep_ranks
        self.ep_node = [node_of[r] for r in ep_ranks]
        self.ep_local = [r - members[node_of[r]][0] for r in ep_ranks]


def _topology(dispatcher) -> _Topology:
    ep_ranks = dist.get_process_group_ranks(dispatcher.ep_group)
    key = (tuple(ep_ranks), dispatcher.tp_size)
    if key not in _TOPOLOGY:
        _TOPOLOGY[key] = _Topology(ep_ranks, dispatcher.tp_size)
    return _TOPOLOGY[key]


def _a2a_counts(counts: torch.Tensor, group) -> torch.Tensor:
    """Exchange per-destination int64 count vectors ([group_size, k] -> [group_size, k])."""
    out = torch.empty_like(counts)
    dist.all_to_all_single(out, counts.contiguous(), group=group)
    return out


def _a2a(group, x: torch.Tensor, send: List[int], recv: List[int]) -> torch.Tensor:
    from megatron.core.tensor_parallel import all_to_all

    return all_to_all(group, x, recv, send)


def _a2a_nograd(group, x: torch.Tensor, send: List[int], recv: List[int]) -> torch.Tensor:
    out = x.new_empty((sum(recv),) + tuple(x.shape[1:]))
    dist.all_to_all_single(out, x.contiguous(), recv, send, group=group)
    return out


def _check_supported(d) -> None:
    cfg = d.config
    problems = []
    if d.drop_and_pad or cfg.moe_expert_capacity_factor is not None:
        problems.append("token dropping")
    if cfg.moe_permute_fusion:
        problems.append("moe_permute_fusion")
    if cfg.moe_router_padding_for_quantization:
        problems.append("moe_router_padding_for_quantization")
    if getattr(cfg, "batch_invariant_mode", False):
        problems.append("batch_invariant_mode")
    if d.shared_experts is not None:
        problems.append("moe_shared_expert_overlap")
    if problems:
        raise NotImplementedError(f"{_ENV}=1 does not support: {', '.join(problems)}")


def _dispatch_preprocess(self, hidden_states, routing_map, probs):
    """Metadata only: megatron-core's ``preprocess``; the per-copy permutation is skipped."""
    _check_supported(self)
    self.hidden_shape = hidden_states.shape
    self.probs = probs
    self.routing_map = routing_map
    hidden_states = hidden_states.view(-1, self.hidden_shape[-1])
    self.tokens_per_expert = self.preprocess(routing_map)
    self.tokens_per_expert = self._maybe_dtoh_and_synchronize("before_permutation_1", self.tokens_per_expert)
    self.hidden_shape_before_permute = hidden_states.shape
    return hidden_states, probs


def _token_dispatch(self, hidden_states, probs):
    topo = _topology(self)
    self.tokens_per_expert = self._maybe_dtoh_and_synchronize("before_ep_alltoall", self.tokens_per_expert)
    routing_map = self.routing_map
    device = hidden_states.device
    num_local = self.num_local_experts
    # Experts hosted by each node for this EP group (contiguous blocks of whole EP ranks).
    node_experts = [[] for _ in range(topo.num_nodes)]
    for e, n in enumerate(topo.ep_node):
        node_experts[n].extend(range(e * num_local, (e + 1) * num_local))
    for n, cols in enumerate(node_experts):
        if cols != list(range(cols[0], cols[0] + len(cols))):
            raise RuntimeError(f"{_ENV}: experts of node {n} are not contiguous")
    width = len(node_experts[0])
    starts = [cols[0] for cols in node_experts]

    # ---- hop 1: one row per (token, destination node), ordered by node then token ----
    tok_idx, bits, prob_rows, send_a = [], [], [], []
    for n in range(topo.num_nodes):
        sub = routing_map[:, starts[n] : starts[n] + width]
        rows = sub.any(dim=1).nonzero(as_tuple=True)[0]
        tok_idx.append(rows)
        bits.append(sub.index_select(0, rows).to(torch.uint8))
        prob_rows.append(probs.index_select(0, rows)[:, starts[n] : starts[n] + width])
    counts_a = torch.tensor([t.numel() for t in tok_idx], dtype=torch.long, device=device)
    send_a = counts_a.tolist()
    recv_a = _a2a_counts(counts_a.view(-1, 1), topo.rail_group).view(-1).tolist()
    idx_a = torch.cat(tok_idx)
    pool_x = _a2a(topo.rail_group, hidden_states.index_select(0, idx_a), send_a, recv_a)
    pool_p = _a2a(topo.rail_group, torch.cat(prob_rows), send_a, recv_a)
    pool_bits = _a2a_nograd(topo.rail_group, torch.cat(bits), send_a, recv_a).bool()
    pool_src_node = torch.repeat_interleave(
        torch.arange(topo.num_nodes, device=device), torch.tensor(recv_a, device=device)
    )

    # ---- hop 2: relay fans out one copy per expert to the expert ranks on this node ----
    # Destination EP ranks on this node (this EP group), in EP order; each gets its rows sorted
    # by (source node, local expert, token), i.e. megatron-core's per-source order.
    # (all_to_all takes the send buffer in group-rank order, i.e. by local index.)
    my_eps = sorted((e for e, n in enumerate(topo.ep_node) if n == topo.node), key=lambda e: topo.ep_local[e])
    base = starts[topo.node]
    copy_rows, copy_probs, send_b = [], [], [0] * topo.node_size
    src_counts = torch.zeros(topo.node_size, topo.num_nodes, dtype=torch.long, device=device)
    for e in my_eps:
        lo = e * num_local - base
        sub = pool_bits[:, lo : lo + num_local]
        row, j = sub.nonzero(as_tuple=True)
        key = pool_src_node[row] * num_local + j
        order = torch.sort(key, stable=True).indices
        row, j = row[order], j[order]
        copy_rows.append(row)
        copy_probs.append(pool_p[row, lo + j])
        dest = topo.ep_local[e]
        send_b[dest] = row.numel()
        src_counts[dest] = torch.bincount(pool_src_node[row], minlength=topo.num_nodes)
    copy_idx = torch.cat(copy_rows) if copy_rows else torch.empty(0, dtype=torch.long, device=device)
    recv_counts = _a2a_counts(src_counts, topo.node_group)  # [relay local idx, source node]
    recv_b = recv_counts.sum(dim=1).tolist()
    x = _a2a(topo.node_group, pool_x.index_select(0, copy_idx), send_b, recv_b)
    p = _a2a(topo.node_group, torch.cat(copy_probs) if copy_probs else pool_p.new_empty(0), send_b, recv_b)

    # ---- reorder blocks (relay, source node) into megatron-core's source order ----
    # The source of block (relay local index l, node n) is the rank (n, l); megatron-core orders
    # sources by their EP rank.
    counts = recv_counts.tolist()
    offsets, off = {}, 0
    for relay in range(topo.node_size):
        for n in range(topo.num_nodes):
            offsets[(relay, n)] = (off, counts[relay][n])
            off += counts[relay][n]
    perm, per_source = [], []
    for e in range(len(topo.ep_ranks)):
        start, cnt = offsets[(topo.ep_local[e], topo.ep_node[e])]
        perm.append(torch.arange(start, start + cnt, device=device))
        per_source.append(cnt)
    perm = torch.cat(perm)
    expected = [int(c) for c in self.output_splits]
    if per_source != expected:
        raise RuntimeError(f"{_ENV}: received {per_source} rows per source, megatron-core expects {expected}")
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel(), device=device)

    self._node_dedup = dict(
        idx_a=idx_a,
        send_a=send_a,
        recv_a=recv_a,
        copy_idx=copy_idx,
        pool_rows=sum(recv_a),
        send_b=send_b,
        recv_b=recv_b,
        inv=inv,
        num_tokens=hidden_states.size(0),
    )
    return x.index_select(0, perm), p.index_select(0, perm)


def _token_combine(self, hidden_states, async_finish: bool = True, allocate_on_comm_stream: bool = True):
    topo = _topology(self)
    st = self._node_dedup
    self._node_dedup = None
    # Back to the (relay, source node) block order, then to the relays.
    x = _a2a(topo.node_group, hidden_states.index_select(0, st["inv"]), st["recv_b"], st["send_b"])
    # Relay: sum each token's copies for this node (bf16, like megatron-core's unpermute).
    partial = torch.zeros(st["pool_rows"], x.size(-1), dtype=x.dtype, device=x.device).index_add(0, st["copy_idx"], x)
    # One partial sum per (token, node) back to the source, which sums them.
    back = _a2a(topo.rail_group, partial, st["recv_a"], st["send_a"])
    return torch.zeros(st["num_tokens"], back.size(-1), dtype=back.dtype, device=back.device).index_add(
        0, st["idx_a"], back
    )


def _combine_postprocess(self, output):
    return output.view(self.hidden_shape)


def patch_moe_node_dedup_dispatch() -> bool:
    """Install the node-deduplicated all-to-all when ``SKYRL_MOE_NODE_DEDUP=1``. Idempotent."""
    global _APPLIED
    if os.environ.get(_ENV, "0").lower() not in ("1", "true"):
        return False
    if _APPLIED:
        return True
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    cls = MoEAlltoAllTokenDispatcher
    cls.dispatch_preprocess = functools.wraps(cls.dispatch_preprocess)(_dispatch_preprocess)
    cls.token_dispatch = functools.wraps(cls.token_dispatch)(_token_dispatch)
    cls.token_combine = functools.wraps(cls.token_combine)(_token_combine)
    cls.combine_postprocess = functools.wraps(cls.combine_postprocess)(_combine_postprocess)
    _APPLIED = True
    logger.info(f"{_ENV}=1: MoE all-to-all sends each token once per destination node")
    return True
