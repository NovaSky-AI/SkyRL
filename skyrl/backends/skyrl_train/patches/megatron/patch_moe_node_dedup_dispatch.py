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

``SKYRL_MOE_NODE_DEDUP=fused`` selects the fused form of the same two hops
(``_moe_twohop_fused``): 2 all-to-alls per dispatch and 2 per combine instead of 7 and 2, no count
exchange after the first all-gather, row kernels that write packed messages in one pass, and a
hand-written backward. It replaces megatron-core's permutation, so it runs with
``moe_permute_fusion`` on or off. It needs ETP 1 in addition to the requirements above.
"""

import contextlib
import functools
import json
import os
import socket
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from loguru import logger

_ENV = "SKYRL_MOE_NODE_DEDUP"
_APPLIED = False

# Opt-in phase timers (SKYRL_MOE_DISPATCH_PROFILE=1), for the stock path and for the two-hop path.
# Every phase is bracketed by torch.cuda.synchronize(), so the numbers split the cost but the run
# is slower than an unprofiled one. Totals go to the log and to
# ``$SKYRL_MOE_DISPATCH_PROFILE_DIR/moe_dispatch_profile_rank{r}.json`` (default /tmp) every
# ``SKYRL_MOE_DISPATCH_PROFILE_EVERY`` combines (default 1000).
_PROFILE_ENV = "SKYRL_MOE_DISPATCH_PROFILE"
_PROFILE_ON = False
_PROFILE: Dict[str, List[float]] = {}  # phase -> [seconds, calls]
_PROFILE_COMBINES = 0


@contextlib.contextmanager
def _phase(name: str):
    if not _PROFILE_ON:
        yield
        return
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        acc = _PROFILE.setdefault(name, [0.0, 0])
        acc[0] += time.perf_counter() - t0
        acc[1] += 1


def _timed(name: str, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _phase(name):
            return fn(*args, **kwargs)

    return wrapper


def _profile_report() -> None:
    """Count one combine; every N of them write the cumulative phase totals."""
    global _PROFILE_COMBINES
    if not _PROFILE_ON:
        return
    _PROFILE_COMBINES += 1
    if _PROFILE_COMBINES % int(os.environ.get("SKYRL_MOE_DISPATCH_PROFILE_EVERY", "1000")):
        return
    rank = dist.get_rank()
    totals = {k: {"seconds": round(v[0], 4), "calls": v[1]} for k, v in sorted(_PROFILE.items())}
    out_dir = os.environ.get("SKYRL_MOE_DISPATCH_PROFILE_DIR", "/tmp")
    with open(os.path.join(out_dir, f"moe_dispatch_profile_rank{rank}.json"), "w") as f:
        json.dump(
            {"rank": rank, "host": socket.gethostname(), "combines": _PROFILE_COMBINES, "phases": totals}, f, indent=1
        )
    lines = "".join(f"\n  {k:<34}{v['seconds']:>10.3f} s {v['calls']:>8} calls" for k, v in totals.items())
    logger.info(f"MoE dispatch profile, rank {rank}, {_PROFILE_COMBINES} combines (cumulative):{lines}")


def _install_profile(mode: str) -> None:
    """Turn the timers on: the dispatcher steps of the active path, the MoE layer forward, and the
    backward of every all-to-all (named by its group size: EP group, node group, or rail group).
    With activation recompute the forward phases also count the recomputed forward."""
    global _PROFILE_ON
    if _PROFILE_ON or os.environ.get(_PROFILE_ENV, "0").lower() not in ("1", "true"):
        return
    from megatron.core.tensor_parallel import mappings
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    _PROFILE_ON = True
    cls = MoEAlltoAllTokenDispatcher
    dedup = mode != "stock"
    # Two-hop path: token_dispatch and token_combine carry their own finer phases. The fused path
    # times every phase itself (forward and backward) and has no work in the other steps.
    steps = [] if mode == "fused" else ["dispatch_preprocess", "dispatch_postprocess", "combine_preprocess"]
    if not dedup:
        steps += ["token_dispatch", "token_combine"]
    for step in steps:
        setattr(cls, step, _timed(f"{mode}/fwd/{step}", getattr(cls, step)))
    if not dedup:
        stock_post = _timed("stock/fwd/combine_postprocess", cls.combine_postprocess)

        @functools.wraps(cls.combine_postprocess)
        def combine_postprocess(self, *args, **kwargs):
            out = stock_post(self, *args, **kwargs)
            _profile_report()
            return out

        cls.combine_postprocess = combine_postprocess
    MoELayer.forward = _timed("moe_layer/fwd_total", MoELayer.forward)

    a2a = mappings._AllToAll
    stock_backward = a2a.backward

    def backward(ctx, *grad_output):
        with _phase(f"bwd/all_to_all/group_of_{ctx.group.size()}"):
            return stock_backward(ctx, *grad_output)

    a2a.backward = staticmethod(backward)
    logger.info(f"{_PROFILE_ENV}=1: MoE dispatch phase timers on ({mode} path)")


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
    if len(send) == 1:
        return x
    out = x.new_empty((sum(recv),) + tuple(x.shape[1:]))
    dist.all_to_all_single(out, x.contiguous(), recv, send, group=group)
    return out


def _check_supported(d, fused: bool = False) -> None:
    cfg = d.config
    problems = []
    if d.drop_and_pad or cfg.moe_expert_capacity_factor is not None:
        problems.append("token dropping")
    if cfg.moe_permute_fusion and not fused:
        problems.append("moe_permute_fusion (the fused mode supports it)")
    if fused and d.tp_size != 1:
        problems.append("expert tensor parallel size > 1 in fused mode")
    if cfg.moe_router_padding_for_quantization:
        problems.append("moe_router_padding_for_quantization")
    if getattr(cfg, "batch_invariant_mode", False):
        problems.append("batch_invariant_mode")
    if d.shared_experts is not None:
        problems.append("moe_shared_expert_overlap")
    if problems:
        raise NotImplementedError(f"{_ENV} does not support: {', '.join(problems)}")


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
    with _phase("twohop/fwd/1_pack"):
        for n in range(topo.num_nodes):
            sub = routing_map[:, starts[n] : starts[n] + width]
            rows = sub.any(dim=1).nonzero(as_tuple=True)[0]
            tok_idx.append(rows)
            bits.append(sub.index_select(0, rows).to(torch.uint8))
            prob_rows.append(probs.index_select(0, rows)[:, starts[n] : starts[n] + width])
        counts_a = torch.tensor([t.numel() for t in tok_idx], dtype=torch.long, device=device)
        send_a = counts_a.tolist()
        idx_a = torch.cat(tok_idx)
        send_x, send_p, send_bits = hidden_states.index_select(0, idx_a), torch.cat(prob_rows), torch.cat(bits)
    # The first collective of a dispatch also absorbs the wait for the slowest peer.
    with _phase("twohop/fwd/2_hop1_counts_and_peer_wait"):
        recv_a = _a2a_counts(counts_a.view(-1, 1), topo.rail_group).view(-1).tolist()
    with _phase("twohop/fwd/3_hop1_network_tokens"):
        pool_x = _a2a(topo.rail_group, send_x, send_a, recv_a)
    with _phase("twohop/fwd/4_hop1_network_probs_bits"):
        pool_p = _a2a(topo.rail_group, send_p, send_a, recv_a)
        pool_bits = _a2a_nograd(topo.rail_group, send_bits, send_a, recv_a).bool()
    with _phase("twohop/fwd/5_expand"):
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
    with _phase("twohop/fwd/6_hop2_counts"):
        recv_counts = _a2a_counts(src_counts, topo.node_group)  # [relay local idx, source node]
        recv_b = recv_counts.sum(dim=1).tolist()
    with _phase("twohop/fwd/7_hop2_nvlink_tokens"):
        x = _a2a(topo.node_group, pool_x.index_select(0, copy_idx), send_b, recv_b)
    with _phase("twohop/fwd/8_hop2_nvlink_probs"):
        p = _a2a(topo.node_group, torch.cat(copy_probs) if copy_probs else pool_p.new_empty(0), send_b, recv_b)

    # ---- reorder blocks (relay, source node) into megatron-core's source order ----
    # The source of block (relay local index l, node n) is the rank (n, l); megatron-core orders
    # sources by their EP rank.
    with _phase("twohop/fwd/9_reorder"):
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
    with _phase("twohop/fwd/a_back_hop2_nvlink"):
        x = _a2a(topo.node_group, hidden_states.index_select(0, st["inv"]), st["recv_b"], st["send_b"])
    # Relay: sum each token's copies for this node (bf16, like megatron-core's unpermute).
    with _phase("twohop/fwd/b_relay_sum"):
        partial = torch.zeros(st["pool_rows"], x.size(-1), dtype=x.dtype, device=x.device).index_add(
            0, st["copy_idx"], x
        )
    # One partial sum per (token, node) back to the source, which sums them.
    with _phase("twohop/fwd/c_back_hop1_network"):
        back = _a2a(topo.rail_group, partial, st["recv_a"], st["send_a"])
    with _phase("twohop/fwd/d_final_sum"):
        out = torch.zeros(st["num_tokens"], back.size(-1), dtype=back.dtype, device=back.device).index_add(
            0, st["idx_a"], back
        )
    _profile_report()
    return out


def _combine_postprocess(self, output):
    return output.view(self.hidden_shape)


# ---------------------------------------------------------------- fused mode
_COMMS: Dict[tuple, object] = {}


def _dist_comm(dispatcher):
    """The fused path's transport for this dispatcher's expert-parallel group (cached)."""
    from skyrl.backends.skyrl_train.patches.megatron import _moe_twohop_fused as fused

    topo = _topology(dispatcher)
    key = (tuple(topo.ep_ranks), dispatcher.tp_size)
    if key in _COMMS:
        return _COMMS[key]
    world = dist.get_world_size()
    world_group = dispatcher.tp_ep_group
    if dist.get_process_group_ranks(world_group) != list(range(world)):
        raise RuntimeError(f"{_ENV}=fused: the expert-parallel group must hold every rank in order")

    class _DistComm(fused.Comm):
        world_size, rank = world, dist.get_rank()
        num_nodes, node_size = topo.num_nodes, topo.node_size

        def all_gather(self, vec):
            out = vec.new_empty(world, vec.numel())
            dist.all_gather_into_tensor(out, vec, group=world_group)
            return out

        def rail_a2a(self, x, send, recv):
            return _a2a_nograd(topo.rail_group, x, send, recv)

        def node_a2a(self, x, send, recv):
            return _a2a_nograd(topo.node_group, x, send, recv)

        def phase(self, name):
            return _phase(f"fused/{name}")

    _COMMS[key] = _DistComm()
    return _COMMS[key]


def _fused_dispatch_preprocess(self, hidden_states, routing_map, probs):
    _check_supported(self, fused=True)
    self.hidden_shape = hidden_states.shape
    self.probs = probs
    self.routing_map = routing_map
    return hidden_states.view(-1, self.hidden_shape[-1]), probs


def _fused_token_dispatch(self, hidden_states, probs):
    from skyrl.backends.skyrl_train.patches.megatron import _moe_twohop_fused as fused

    self._twohop_state = fused.TwoHopState(_dist_comm(self), self.num_local_experts, self.config.moe_router_topk)
    return fused.dispatch(self._twohop_state, hidden_states, probs, self.routing_map)


def _fused_dispatch_postprocess(self, hidden_states, probs):
    # Rows already arrive ordered by local expert.
    return hidden_states, self._twohop_state.tokens_per_expert, probs


def _fused_combine_preprocess(self, hidden_states):
    return hidden_states


def _fused_token_combine(self, hidden_states, async_finish: bool = True, allocate_on_comm_stream: bool = True):
    from skyrl.backends.skyrl_train.patches.megatron import _moe_twohop_fused as fused

    state, self._twohop_state = self._twohop_state, None
    out = fused.combine(state, hidden_states)
    _profile_report()
    return out


_STEPS = (
    "dispatch_preprocess",
    "token_dispatch",
    "dispatch_postprocess",
    "combine_preprocess",
    "token_combine",
    "combine_postprocess",
)


def patch_moe_node_dedup_dispatch() -> bool:
    """Install the node-deduplicated all-to-all when ``SKYRL_MOE_NODE_DEDUP`` is ``1`` (tensor ops)
    or ``fused`` (row kernels, 2 all-to-alls per direction). Idempotent."""
    global _APPLIED
    mode = os.environ.get(_ENV, "0").lower()
    if mode not in ("1", "true", "fused"):
        _install_profile("stock")
        return False
    if _APPLIED:
        return True
    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAlltoAllTokenDispatcher,
    )

    cls = MoEAlltoAllTokenDispatcher
    if mode == "fused":
        new = dict(
            dispatch_preprocess=_fused_dispatch_preprocess,
            token_dispatch=_fused_token_dispatch,
            dispatch_postprocess=_fused_dispatch_postprocess,
            combine_preprocess=_fused_combine_preprocess,
            token_combine=_fused_token_combine,
            combine_postprocess=_combine_postprocess,
        )
    else:
        new = dict(
            dispatch_preprocess=_dispatch_preprocess,
            token_dispatch=_token_dispatch,
            token_combine=_token_combine,
            combine_postprocess=_combine_postprocess,
        )
    for step, fn in new.items():
        setattr(cls, step, functools.wraps(getattr(cls, step))(fn))
    _APPLIED = True
    _install_profile("fused" if mode == "fused" else "twohop")
    logger.info(
        f"{_ENV}={mode}: MoE all-to-all sends each token once per destination node"
        + (" (fused kernels)" if mode == "fused" else "")
    )
    return True
