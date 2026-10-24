"""Measure NCCL all-to-all bandwidth across Ray nodes and report the transport NCCL picked.

    python nccl_a2a_probe.py <gpus_per_node> <num_nodes> [MiB per rank]

Starts one actor per GPU (``gpus_per_node`` on each of ``num_nodes`` distinct nodes), runs
``all_to_all_single`` and prints the per-rank bandwidth. NCCL_DEBUG=INFO output (INIT,NET) goes to
the Ray worker logs, which the driver echoes: look for ``NET/OFI`` with the efa provider (RDMA)
versus ``NET/Socket`` (TCP fallback). Any NCCL_* variables set for the driver are forwarded.
"""

import os
import sys
import time

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy


@ray.remote(num_gpus=1)
class _Rank:
    def addr(self):
        import socket

        from ray.util import get_node_ip_address

        s = socket.socket()
        s.bind(("", 0))
        return get_node_ip_address(), s.getsockname()[1]

    def run(self, rank, world, addr, port, mib, env):
        os.environ.update(env)
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_SUBSYS"] = "INIT,NET"
        import torch
        import torch.distributed as dist

        torch.cuda.set_device(0)
        dist.init_process_group("nccl", init_method=f"tcp://{addr}:{port}", rank=rank, world_size=world)
        x = torch.randn(mib * 2**20 // 2, device="cuda", dtype=torch.bfloat16)
        y = torch.empty_like(x)
        for _ in range(3):
            dist.all_to_all_single(y, x)
        torch.cuda.synchronize()
        iters = 10
        t = time.perf_counter()
        for _ in range(iters):
            dist.all_to_all_single(y, x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / iters
        dist.destroy_process_group()
        return {
            "node": ray.util.get_node_ip_address(),
            "ms": dt * 1e3,
            "GBps_per_rank": x.numel() * 2 * (world - 1) / world / dt / 1e9,  # bytes sent to other ranks
        }


def main():
    gpn, nodes = int(sys.argv[1]), int(sys.argv[2])
    mib = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    ray.init(address="auto", log_to_driver=True)
    pg = placement_group([{"GPU": gpn, "CPU": gpn}] * nodes, strategy="STRICT_SPREAD")
    ray.get(pg.ready(), timeout=120)
    ranks = [
        _Rank.options(
            scheduling_strategy=PlacementGroupSchedulingStrategy(placement_group=pg, placement_group_bundle_index=b)
        ).remote()
        for b in range(nodes)
        for _ in range(gpn)
    ]
    addr, port = ray.get(ranks[0].addr.remote())
    env = {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "FI_")) and k != "NCCL_DEBUG"}
    res = ray.get([r.run.remote(i, len(ranks), addr, port, mib, env) for i, r in enumerate(ranks)])
    print(
        f"RESULT world={len(ranks)} nodes={len({r['node'] for r in res})} msg={mib}MiB/rank "
        f"time={max(r['ms'] for r in res):.1f}ms min_per_rank_bw={min(r['GBps_per_rank'] for r in res):.1f}GB/s"
    )


if __name__ == "__main__":
    main()
