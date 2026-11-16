"""Replay expert-parallel all-to-all traffic and time it. No model; payload is random.

Input: ``routes.npz`` from ``extract_routes.py``. One process per EP rank:

    # one node, 8 GPUs
    uv run --isolated --extra megatron torchrun --nproc-per-node 8 a2a_bench.py routes.npz \
        --hidden-size 2048 --gpus-per-node 8 --out result.json

    # two nodes: run on each node, with the same rendezvous endpoint
    ... torchrun --nnodes 2 --nproc-per-node 8 --rdzv-endpoint HOST:PORT a2a_bench.py ...

Every all-to-all call in the file is replayed as a dispatch (matrix M) and a combine (M transposed),
which is what one MoE layer does in the forward pass. The backward pass repeats both.

Backends (``--backends``), all moving the same bytes:
  stock        ``torch.distributed.all_to_all_single``: the call Megatron's alltoall dispatcher makes.
  p2p_group    one send and one receive per peer, all issued inside one NCCL group.
  p2p_rounds   pairwise exchanges in ``ep - 1`` rounds (round-robin tournament): in each round
               every rank talks to exactly one peer.

Traffic variants (``--variants``):
  real         the matrices as recorded.
  uniform      every entry replaced by the matrix mean: same bytes, no skew.
  intra        only entries whose two ranks share a node (NVLink).
  cross        only entries whose two ranks are on different nodes (network).

Each rank times each transfer between two ``torch.cuda.synchronize()`` calls, after a barrier so
that all ranks start together. The time of a transfer is the slowest rank's time.
``--backend-kind gloo --device cpu`` runs the p2p backends on CPU, to test the script only.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist


def variant_matrix(m: np.ndarray, variant: str, gpus_per_node: int) -> np.ndarray:
    ep = m.shape[0]
    node = np.arange(ep) // gpus_per_node
    same_node = node[:, None] == node[None, :]
    if variant == "real":
        return m
    if variant == "uniform":
        return np.full_like(m, int(round(m.mean())))
    if variant == "intra":
        return np.where(same_node, m, 0)
    if variant == "cross":
        return np.where(same_node, 0, m)
    raise ValueError(variant)


def round_robin_rounds(n: int):
    """Circle method: n - 1 rounds (n even), every pair meets once. rounds[r][i] = peer of rank i."""
    assert n % 2 == 0, "p2p_rounds needs an even EP size"
    ids = list(range(n))
    rounds = []
    for _ in range(n - 1):
        peer = [0] * n
        for i in range(n // 2):
            a, b = ids[i], ids[n - 1 - i]
            peer[a], peer[b] = b, a
        rounds.append(peer)
        ids = [ids[0]] + [ids[-1]] + ids[1:-1]
    return rounds


class Transfer:
    """One all-to-all with send counts ``send[dst]`` and receive counts ``recv[src]`` for this rank."""

    def __init__(self, rank, ep, hidden, dtype, device, max_rows):
        self.rank, self.ep, self.hidden = rank, ep, hidden
        self.src = torch.randn(max_rows, hidden, dtype=dtype, device=device)
        self.dst = torch.empty(max_rows, hidden, dtype=dtype, device=device)
        self.rounds = round_robin_rounds(ep) if ep % 2 == 0 else None

    def _chunks(self, buf, counts):
        offs = np.concatenate([[0], np.cumsum(counts)])
        return [buf[offs[i] : offs[i + 1]] for i in range(self.ep)]

    def stock(self, send, recv):
        dist.all_to_all_single(
            self.dst[: int(recv.sum())],
            self.src[: int(send.sum())],
            output_split_sizes=recv.tolist(),
            input_split_sizes=send.tolist(),
        )

    def _exchange(self, peers, out, inp):
        ops = []
        for p in peers:
            if p == self.rank:
                continue
            if inp[p].shape[0]:
                ops.append(dist.P2POp(dist.irecv, inp[p], p))
            if out[p].shape[0]:
                ops.append(dist.P2POp(dist.isend, out[p], p))
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()

    def p2p_group(self, send, recv):
        out, inp = self._chunks(self.src, send), self._chunks(self.dst, recv)
        inp[self.rank].copy_(out[self.rank])  # rows that stay on this rank
        self._exchange(range(self.ep), out, inp)

    def p2p_rounds(self, send, recv):
        out, inp = self._chunks(self.src, send), self._chunks(self.dst, recv)
        inp[self.rank].copy_(out[self.rank])
        for peer in self.rounds:
            self._exchange([peer[self.rank]], out, inp)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("routes")
    p.add_argument("--hidden-size", type=int, required=True)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--backends", default="stock,p2p_group,p2p_rounds")
    p.add_argument("--variants", default="real,uniform,intra,cross")
    p.add_argument("--max-calls", type=int, default=0, help="replay only the first N calls (0 = all)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--backend-kind", default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    dist.init_process_group(a.backend_kind)
    rank, world = dist.get_rank(), dist.get_world_size()
    cuda = a.device == "cuda"
    if cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % a.gpus_per_node)))
    device = torch.device(a.device)
    dtype = getattr(torch, a.dtype)

    data = np.load(a.routes)
    matrices = data["matrices"].astype(np.int64)
    if a.max_calls:
        matrices = matrices[: a.max_calls]
    ep = matrices.shape[1]
    assert world == ep, f"launch {ep} processes (one per EP rank); got {world}"
    row_bytes = a.hidden_size * torch.empty(0, dtype=dtype).element_size()

    max_rows = int(max(matrices.sum(axis=2).max(), matrices.sum(axis=1).max()))
    xfer = Transfer(rank, ep, a.hidden_size, dtype, device, max_rows)
    sync = torch.cuda.synchronize if cuda else (lambda: None)

    def timed(fn, send, recv):
        dist.barrier()
        sync()
        t0 = time.perf_counter()
        fn(send, recv)
        sync()
        return time.perf_counter() - t0

    results = []
    for backend in a.backends.split(","):
        fn = getattr(xfer, backend)
        for variant in a.variants.split(","):
            mats = [variant_matrix(m, variant, a.gpus_per_node) for m in matrices]
            if not any(m.sum() - np.trace(m) for m in mats):
                continue  # e.g. "cross" on a single node
            for m in mats[: a.warmup]:
                timed(fn, m[rank], m[:, rank])
            times = torch.zeros(len(mats), 2, dtype=torch.float64)
            for i, m in enumerate(mats):
                times[i, 0] = timed(fn, m[rank], m[:, rank])  # dispatch
                times[i, 1] = timed(fn, m[:, rank], m[rank])  # combine: the transposed matrix
            dist.all_reduce(times, op=dist.ReduceOp.MAX)  # slowest rank per transfer
            if rank == 0:
                offdiag = np.array([m.sum() - np.trace(m) for m in mats], dtype=np.float64)
                gb = 2 * offdiag.sum() * row_bytes / 1e9  # dispatch + combine, rows that change rank
                secs = float(times.sum())
                # busiest rank: bytes it sends plus bytes it receives, per transfer
                per_rank = np.array([(m.sum(axis=0) + m.sum(axis=1) - 2 * np.diag(m)).max() for m in mats])
                busiest_gbps = float((per_rank * row_bytes / 1e9 / times.mean(dim=1).numpy()).mean())
                results.append(
                    dict(
                        backend=backend,
                        variant=variant,
                        calls=len(mats),
                        seconds_forward=secs,
                        seconds_forward_and_backward=2 * secs,
                        gigabytes_forward=gb,
                        aggregate_gb_per_s=gb / secs,
                        busiest_rank_gb_per_s=busiest_gbps,
                        ms_per_transfer_p50=float(times.median()) * 1e3,
                        ms_per_transfer_max=float(times.max()) * 1e3,
                    )
                )

    if rank == 0:
        print(f"ep {ep}, hidden {a.hidden_size}, {a.dtype}, {len(matrices)} calls, {a.gpus_per_node} GPUs per node")
        print(
            f"{'backend':<11}{'variant':<9}{'fwd s':>9}{'GB':>10}{'agg GB/s':>10}{'busiest GB/s':>14}{'p50 ms':>9}{'max ms':>9}"
        )
        for r in results:
            print(
                f"{r['backend']:<11}{r['variant']:<9}{r['seconds_forward']:>9.3f}{r['gigabytes_forward']:>10.1f}"
                f"{r['aggregate_gb_per_s']:>10.1f}{r['busiest_rank_gb_per_s']:>14.1f}"
                f"{r['ms_per_transfer_p50']:>9.2f}{r['ms_per_transfer_max']:>9.2f}"
            )
        if a.out:
            with open(a.out, "w") as f:
                json.dump(dict(args=vars(a), results=results), f, indent=1)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
