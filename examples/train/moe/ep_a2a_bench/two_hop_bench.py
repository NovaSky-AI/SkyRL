"""Check and time ``DirectDispatch`` against ``TwoHopDispatch`` (see ``two_hop.py``). No model.

One process per EP rank. Routes are random (``--synthetic-tokens``) or replayed from a SkyRL batch
dump (``--dump``, written by ``trainer.dump_data_batch=true`` with router replay on):

    # correctness on CPU, 8 processes as 2 pseudo-nodes of 4
    torchrun --nproc-per-node 8 two_hop_bench.py --synthetic-tokens 2000 --num-experts 64 --topk 4 \
        --hidden-size 32 --gpus-per-node 4 --backend-kind gloo --device cpu --dtype float32 --transport p2p

    # timing on two 8-GPU nodes, real routes (run on each node with the same rendezvous endpoint)
    torchrun --nnodes 2 --nproc-per-node 8 --rdzv-endpoint HOST:PORT two_hop_bench.py --dump DUMP.pkl \
        --dp 16 --num-experts 512 --max-tokens-per-microbatch 10752 --hidden-size 2048 --layers 0,1

Every call is checked: the rows each expert receives (count and sum, against an all-reduce of what
the sources hold) and the combined output (against ``sum_k p * expert(x)`` computed locally; the
stand-in expert scales its input by ``1 + id / num_experts``).
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist
from two_hop import Comm, DirectDispatch, TwoHopDispatch


def synthetic_calls(a, rank):
    g = torch.Generator().manual_seed(1234 + rank)
    for _ in range(a.calls):
        t = int(a.synthetic_tokens * (0.5 + torch.rand(1, generator=g).item()))  # uneven tokens per rank
        scores = torch.rand(t, a.num_experts, generator=g)
        yield scores.topk(a.topk, dim=1).indices


def dump_calls(a, rank, ep):
    from moe_imbalance import contiguous_split, load_batch, microbatch_slots

    lengths, values, cu, bounds = load_batch(a.dump)
    layers = [int(x) for x in a.layers.split(",")]
    n = 0
    for lo, hi in bounds:
        rank_ids = contiguous_split(np.arange(lo, hi, dtype=np.int64), a.dp)
        for slot in microbatch_slots(rank_ids, lengths, a.max_tokens_per_microbatch):
            ids = slot[rank]  # EP group 0: DP ranks 0..ep-1
            for layer in layers:
                if a.calls and n >= a.calls:
                    return
                n += 1
                segs = [values[cu[s] : cu[s + 1], layer] for s in ids]
                k = values.shape[2]
                yield torch.from_numpy(np.concatenate(segs).astype(np.int64) if segs else np.zeros((0, k), np.int64))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump")
    p.add_argument("--dp", type=int)
    p.add_argument("--max-tokens-per-microbatch", type=int)
    p.add_argument("--layers", default="0")
    p.add_argument("--synthetic-tokens", type=int, default=0, help="mean tokens per rank per call")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--calls", type=int, default=4, help="calls to run (with --dump: 0 = all)")
    p.add_argument("--num-experts", type=int, required=True)
    p.add_argument("--hidden-size", type=int, required=True)
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--transport", default="a2a", choices=["a2a", "p2p"])
    p.add_argument("--backend-kind", default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--rtol", type=float, default=2e-2)
    p.add_argument("--out")
    a = p.parse_args(argv)

    dist.init_process_group(a.backend_kind)
    rank, ep = dist.get_rank(), dist.get_world_size()
    if a.device == "cuda":
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % a.gpus_per_node)))
    device, dtype = torch.device(a.device), getattr(torch, a.dtype)
    comm = Comm(ep, a.gpus_per_node, a.transport, device)
    calls = list(synthetic_calls(a, rank) if a.synthetic_tokens else dump_calls(a, rank, ep))
    topk = calls[0].shape[1]
    methods = {
        "direct": DirectDispatch(comm, a.num_experts, a.hidden_size, topk, dtype),
        "twohop": TwoHopDispatch(comm, a.num_experts, a.hidden_size, topk, dtype),
    }
    epr = a.num_experts // ep
    scale = (1 + torch.arange(epr, device=device) / a.num_experts + rank * epr / a.num_experts).to(dtype)

    g = torch.Generator().manual_seed(99 + rank)
    times = torch.zeros(len(calls), len(methods), 2, dtype=torch.float64)
    worst = torch.zeros(2, dtype=torch.float64)  # worst relative error: expert input sums, combined output
    for i, idx in enumerate([calls[0]] * a.warmup + calls):
        measured = i >= a.warmup
        if i == a.warmup:
            comm.timers.clear()
        idx = idx.to(device)
        x = torch.randn(idx.shape[0], a.hidden_size, generator=g).to(device=device, dtype=dtype)
        probs = torch.rand(idx.shape, generator=g).to(device)
        probs = probs / probs.sum(dim=1, keepdim=True)

        # Oracles. Expected rows and row-sum per expert: all-reduce of what each source holds.
        want = torch.zeros(a.num_experts, a.hidden_size + 1, dtype=torch.float64, device=device)
        flat = idx.reshape(-1)
        want[:, 0].index_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
        want[:, 1:].index_add_(0, flat, x.double().repeat_interleave(topk, dim=0))
        dist.all_reduce(want)
        want = want[rank * epr : (rank + 1) * epr]
        ref_out = (x.double().unsqueeze(1) * (probs.double() * (1 + idx / a.num_experts)).unsqueeze(2)).sum(dim=1)

        for m, (name, d) in enumerate(methods.items()):
            dist.barrier()
            comm.sync()
            t0 = time.perf_counter()
            rows, per_expert = d.dispatch(x, idx, probs)
            comm.sync()
            t1 = time.perf_counter()
            assert torch.equal(per_expert.double(), want[:, 0]), f"{name}: wrong row count per expert"
            owner = torch.repeat_interleave(torch.arange(epr, device=device), per_expert)
            got = torch.zeros(epr, a.hidden_size, dtype=torch.float64, device=device).index_add_(
                0, owner, rows.double()
            )
            err_in = (got - want[:, 1:]).abs().max() / want[:, 1:].abs().max().clamp_min(1e-9)
            y = rows * scale[owner].unsqueeze(1)  # stand-in expert
            dist.barrier()
            comm.sync()
            t2 = time.perf_counter()
            out = d.combine(y)
            comm.sync()
            t3 = time.perf_counter()
            err_out = (out.double() - ref_out).abs().max() / ref_out.abs().max().clamp_min(1e-9)
            worst = torch.maximum(worst, torch.stack([err_in, err_out]).cpu())
            if measured:
                times[i - a.warmup, m, 0], times[i - a.warmup, m, 1] = t1 - t0, t3 - t2

    dist.all_reduce(times, op=dist.ReduceOp.MAX)  # slowest rank per call
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    names = sorted(comm.timers)
    phases = torch.tensor([comm.timers[n] for n in names], dtype=torch.float64)
    dist.all_reduce(phases, op=dist.ReduceOp.MAX)
    if rank == 0:
        ok = bool((worst < a.rtol).all())
        print(
            f"ep {ep}, {a.gpus_per_node} per node, experts {a.num_experts}, top-{topk}, hidden {a.hidden_size}, {a.dtype}, {len(calls)} calls"
        )
        print(
            f"check: expert-input rel err {worst[0]:.2e}, output rel err {worst[1]:.2e} -> {'PASS' if ok else 'FAIL'}"
        )
        total = times.sum(dim=0)
        for m, name in enumerate(methods):
            print(f"{name:<8} dispatch {total[m, 0]:.4f} s   combine {total[m, 1]:.4f} s   sum {total[m].sum():.4f} s")
        print("phases (slowest rank per phase, summed over calls):")
        for n, v in zip(names, phases.tolist()):
            print(f"  {n:<28}{v:.4f} s")
        if a.out:
            with open(a.out, "w") as f:
                json.dump(
                    dict(
                        args=vars(a),
                        check_pass=ok,
                        worst_rel_err=worst.tolist(),
                        seconds=total.tolist(),
                        phases=dict(zip(names, phases.tolist())),
                    ),
                    f,
                    indent=1,
                )
        assert ok, "correctness check failed"
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
