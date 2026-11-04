"""NCCL bandwidth across the Ray cluster: one actor per GPU, ranks ordered by (node, GPU).

Reports bus bandwidth (nccl-tests convention) for all_reduce within node 0, all_reduce and all_to_all over every
GPU, and send/recv between GPU 0 of the first two nodes.
"""

import os
import socket
import time

import ray


@ray.remote(num_gpus=1)
class Worker:
    def info(self):
        return ray.util.get_node_ip_address(), int(ray.get_gpu_ids()[0])

    def free_port(self):
        s = socket.socket()
        s.bind(("", 0))
        return s.getsockname()[1]

    def run(self, rank, world, master, port, node_ranks):
        import torch
        import torch.distributed as dist

        os.environ.setdefault("NCCL_DEBUG", "INFO")
        os.environ.setdefault("NCCL_DEBUG_SUBSYS", "NET")
        torch.cuda.set_device(0)
        dist.init_process_group("nccl", init_method=f"tcp://{master}:{port}", rank=rank, world_size=world)
        out = {}
        if rank == 0:
            out["device"] = torch.cuda.get_device_name(0)
            out["cap"] = torch.cuda.get_device_capability(0)
            out["mem_gb"] = torch.cuda.get_device_properties(0).total_memory / 2**30
            out["env"] = {k: v for k, v in os.environ.items() if k.startswith(("NCCL_", "FI_"))}

        def timed(fn, iters=20, warmup=5, group=None):
            for _ in range(warmup):
                fn()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            t = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t) / iters

        nbytes = 1 << 30
        x = torch.ones(nbytes // 2, dtype=torch.bfloat16, device="cuda")

        groups = [dist.new_group(r) for r in node_ranks]
        my_node = next(i for i, r in enumerate(node_ranks) if rank in r)
        g = groups[0]
        if my_node == 0:
            n = len(node_ranks[0])
            dt = timed(lambda: dist.all_reduce(x, group=g), group=g)
            out["allreduce_intra_busbw_GBps"] = nbytes * 2 * (n - 1) / n / dt / 1e9
        dist.barrier()

        dt = timed(lambda: dist.all_reduce(x))
        out["allreduce_global_busbw_GBps"] = nbytes * 2 * (world - 1) / world / dt / 1e9

        y = torch.empty_like(x)
        dt = timed(lambda: dist.all_to_all_single(y, x))
        out["alltoall_global_busbw_GBps"] = nbytes * (world - 1) / world / dt / 1e9

        if len(node_ranks) > 1:
            a, b = node_ranks[0][0], node_ranks[1][0]
            if rank in (a, b):
                peer = b if rank == a else a

                def p2p():
                    ops = [dist.P2POp(dist.isend, x, peer), dist.P2POp(dist.irecv, y, peer)]
                    for req in dist.batch_isend_irecv(ops):
                        req.wait()

                for _ in range(5):
                    p2p()
                torch.cuda.synchronize()
                t = time.perf_counter()
                for _ in range(20):
                    p2p()
                torch.cuda.synchronize()
                out["sendrecv_internode_GBps_each_way"] = nbytes / ((time.perf_counter() - t) / 20) / 1e9
        dist.barrier()
        dist.destroy_process_group()
        return rank, out


if __name__ == "__main__":
    ray.init()
    num_gpus = int(ray.cluster_resources()["GPU"])
    workers = [Worker.remote() for _ in range(num_gpus)]
    infos = ray.get([w.info.remote() for w in workers])
    order = sorted(range(num_gpus), key=lambda i: infos[i])
    workers = [workers[i] for i in order]
    infos = [infos[i] for i in order]
    ips = sorted({ip for ip, _ in infos})
    node_ranks = [[r for r, (ip, _) in enumerate(infos) if ip == node] for node in ips]
    print(f"{num_gpus} GPUs on {len(ips)} nodes: {node_ranks}")
    master = infos[0][0]
    port = ray.get(workers[0].free_port.remote())
    results = ray.get([w.run.remote(r, num_gpus, master, port, node_ranks) for r, w in enumerate(workers)])
    merged = {}
    for _, out in sorted(results):
        for k, v in out.items():
            merged.setdefault(k, v)
    for k, v in merged.items():
        print(f"RESULT {k}: {round(v, 1) if isinstance(v, float) else v}")
