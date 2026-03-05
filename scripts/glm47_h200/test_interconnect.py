"""
Multi-node interconnect bandwidth test using NCCL all-reduce.

Tests GPU-to-GPU communication bandwidth across nodes to determine if
EFA/InfiniBand is active or if traffic is going over TCP/Ethernet.

Expected bandwidth:
- EFA (AWS): ~100-400 Gbps (12-50 GB/s) for all-reduce
- InfiniBand HDR: ~200 Gbps (25 GB/s)
- TCP/Ethernet (25Gbps NIC): ~3 GB/s
- TCP/Ethernet (10Gbps NIC): ~1.2 GB/s

Usage:
    RAY_ADDRESS=auto python scripts/glm47_h200/test_interconnect.py
"""
import os
import sys
import time
import json

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_gpus=1)
def run_nccl_benchmark(rank, world_size, master_addr, master_port, data_sizes_mb):
    """Run NCCL all-reduce benchmark on a single GPU."""
    import os
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

    # Initialize process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device("cuda:0")

    results = []
    warmup_iters = 3
    bench_iters = 10

    for size_mb in data_sizes_mb:
        num_elements = int(size_mb * 1024 * 1024 / 4)  # float32
        tensor = torch.randn(num_elements, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(warmup_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(bench_iters):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / bench_iters
        # all-reduce sends 2*(n-1)/n * data_size bytes (ring algorithm)
        algo_bw = (size_mb / 1024) / avg_time  # GB/s
        bus_bw = algo_bw * 2 * (world_size - 1) / world_size  # bus bandwidth

        results.append({
            "size_mb": size_mb,
            "avg_time_ms": round(avg_time * 1000, 2),
            "algo_bw_gbps": round(algo_bw, 2),
            "bus_bw_gbps": round(bus_bw, 2),
        })

        if rank == 0:
            print(f"  {size_mb:>6} MB | {avg_time*1000:>8.2f} ms | "
                  f"algo={algo_bw:>6.2f} GB/s | bus={bus_bw:>6.2f} GB/s",
                  flush=True)

    dist.destroy_process_group()
    return results if rank == 0 else None


@ray.remote(num_gpus=1)
def run_intra_node_p2p(rank, peer_rank, master_addr, master_port, data_sizes_mb):
    """Run point-to-point send/recv between two GPUs on the same node."""
    import os
    import torch
    import torch.distributed as dist

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"

    dist.init_process_group(backend="nccl", rank=rank, world_size=2)
    device = torch.device("cuda:0")

    results = []
    warmup_iters = 3
    bench_iters = 10

    for size_mb in data_sizes_mb:
        num_elements = int(size_mb * 1024 * 1024 / 4)
        tensor = torch.randn(num_elements, device=device, dtype=torch.float32)

        # Warmup
        for _ in range(warmup_iters):
            if rank == 0:
                dist.send(tensor, dst=1)
            else:
                dist.recv(tensor, src=0)
        torch.cuda.synchronize()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(bench_iters):
            if rank == 0:
                dist.send(tensor, dst=1)
            else:
                dist.recv(tensor, src=0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / bench_iters
        bw = (size_mb / 1024) / avg_time

        results.append({
            "size_mb": size_mb,
            "avg_time_ms": round(avg_time * 1000, 2),
            "bw_gbps": round(bw, 2),
        })

        if rank == 0:
            print(f"  {size_mb:>6} MB | {avg_time*1000:>8.2f} ms | bw={bw:>6.2f} GB/s",
                  flush=True)

    dist.destroy_process_group()
    return results if rank == 0 else None


def get_free_port():
    """Get a free port on the current machine."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def main():
    ray.init(address="auto")

    gpu_nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
    gpu_nodes.sort(key=lambda n: n["NodeManagerAddress"])

    print(f"Found {len(gpu_nodes)} GPU nodes:")
    for n in gpu_nodes:
        ip = n["NodeManagerAddress"]
        gpus = int(n["Resources"].get("GPU", 0))
        print(f"  {ip} ({gpus} GPUs)")

    if len(gpu_nodes) < 2:
        print("\nNeed at least 2 GPU nodes for cross-node tests.")
        print("Running intra-node test only.\n")

    data_sizes_mb = [1, 10, 100, 512, 1024, 2048]

    all_results = {}

    # =========================================================================
    # Test 1: Intra-node GPU-to-GPU (NVLink)
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 1: Intra-node GPU P2P (NVLink bandwidth)")
    print(f"{'='*70}")

    node0 = gpu_nodes[0]
    node0_ip = node0["NodeManagerAddress"]
    sched0 = NodeAffinitySchedulingStrategy(node_id=node0["NodeID"], soft=False)
    port = 29500

    print(f"Node: {node0_ip}, GPU 0 <-> GPU 1")
    print(f"  {'Size':>6} | {'Time':>10} | {'Bandwidth':>12}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*12}")

    refs = [
        run_intra_node_p2p.options(scheduling_strategy=sched0).remote(
            0, 1, node0_ip, port, data_sizes_mb),
        run_intra_node_p2p.options(scheduling_strategy=sched0).remote(
            1, 0, node0_ip, port, data_sizes_mb),
    ]
    results = ray.get(refs)
    intra_results = [r for r in results if r is not None][0]
    all_results["intra_node_p2p"] = intra_results

    if len(gpu_nodes) >= 2:
        # =====================================================================
        # Test 2: Cross-node 2-GPU all-reduce (one GPU per node)
        # =====================================================================
        print(f"\n{'='*70}")
        print("TEST 2: Cross-node all-reduce (1 GPU per node, 2 nodes)")
        print(f"{'='*70}")

        node1 = gpu_nodes[1]
        node1_ip = node1["NodeManagerAddress"]
        sched1 = NodeAffinitySchedulingStrategy(node_id=node1["NodeID"], soft=False)
        port = 29501

        print(f"Nodes: {node0_ip} <-> {node1_ip}")
        print(f"  {'Size':>6} | {'Time':>10} | {'Algo BW':>10} | {'Bus BW':>10}")
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        refs = [
            run_nccl_benchmark.options(scheduling_strategy=sched0).remote(
                0, 2, node0_ip, port, data_sizes_mb),
            run_nccl_benchmark.options(scheduling_strategy=sched1).remote(
                1, 2, node0_ip, port, data_sizes_mb),
        ]
        results = ray.get(refs, timeout=300)
        cross_2gpu = [r for r in results if r is not None][0]
        all_results["cross_node_2gpu_allreduce"] = cross_2gpu

        # =====================================================================
        # Test 3: Cross-node 8-GPU all-reduce (4 GPUs per node, 2 nodes)
        # =====================================================================
        print(f"\n{'='*70}")
        print("TEST 3: Cross-node all-reduce (4 GPUs per node, 2 nodes = 8 GPUs)")
        print(f"{'='*70}")

        port = 29502
        print(f"Nodes: {node0_ip}(4 GPUs) <-> {node1_ip}(4 GPUs)")
        print(f"  {'Size':>6} | {'Time':>10} | {'Algo BW':>10} | {'Bus BW':>10}")
        print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        refs = []
        for rank in range(8):
            node_idx = rank // 4
            sched = sched0 if node_idx == 0 else sched1
            refs.append(
                run_nccl_benchmark.options(scheduling_strategy=sched).remote(
                    rank, 8, node0_ip, port, data_sizes_mb)
            )
        results = ray.get(refs, timeout=300)
        cross_8gpu = [r for r in results if r is not None][0]
        all_results["cross_node_8gpu_allreduce"] = cross_8gpu

    # =========================================================================
    # Test 4: Check EFA / network interfaces
    # =========================================================================
    print(f"\n{'='*70}")
    print("TEST 4: Network interface check")
    print(f"{'='*70}")

    @ray.remote(num_cpus=0.1)
    def check_network():
        import subprocess
        import os

        info = {"hostname": os.uname().nodename}

        # Check for EFA devices
        try:
            result = subprocess.run(["fi_info", "-p", "efa"], capture_output=True, text=True, timeout=5)
            info["efa_available"] = result.returncode == 0
            if result.returncode == 0:
                info["efa_devices"] = result.stdout.count("provider: efa")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            info["efa_available"] = False

        # Check for InfiniBand
        try:
            result = subprocess.run(["ibstat"], capture_output=True, text=True, timeout=5)
            info["ib_available"] = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            info["ib_available"] = False

        # Check NCCL env vars
        nccl_vars = {k: v for k, v in os.environ.items() if "NCCL" in k}
        info["nccl_env_vars"] = nccl_vars

        # Check network interfaces
        try:
            result = subprocess.run(["ip", "link", "show"], capture_output=True, text=True, timeout=5)
            interfaces = [line.split(":")[1].strip() for line in result.stdout.splitlines()
                         if ": " in line and "@" not in line and "lo" not in line]
            info["network_interfaces"] = interfaces[:10]
        except Exception:
            pass

        return info

    for node in gpu_nodes[:2]:
        sched = NodeAffinitySchedulingStrategy(node_id=node["NodeID"], soft=False)
        net_info = ray.get(check_network.options(scheduling_strategy=sched).remote(), timeout=30)
        ip = node["NodeManagerAddress"]
        print(f"\n  Node {ip} ({net_info.get('hostname', '?')}):")
        print(f"    EFA available: {net_info.get('efa_available', 'unknown')}")
        if net_info.get("efa_available"):
            print(f"    EFA devices: {net_info.get('efa_devices', '?')}")
        print(f"    InfiniBand available: {net_info.get('ib_available', 'unknown')}")
        print(f"    Network interfaces: {net_info.get('network_interfaces', [])}")
        nccl_vars = net_info.get("nccl_env_vars", {})
        if nccl_vars:
            print(f"    NCCL env vars: {json.dumps(nccl_vars, indent=6)}")
        else:
            print(f"    NCCL env vars: (none set)")

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n\n{'='*70}")
    print("INTERCONNECT SUMMARY")
    print(f"{'='*70}")

    if "intra_node_p2p" in all_results:
        best = max(all_results["intra_node_p2p"], key=lambda x: x["bw_gbps"])
        print(f"  Intra-node P2P (NVLink):     {best['bw_gbps']:>6.1f} GB/s @ {best['size_mb']} MB")
        if best["bw_gbps"] > 100:
            print(f"    -> NVLink 4.0 (H200): expected ~450 GB/s bidirectional")
        elif best["bw_gbps"] > 30:
            print(f"    -> NVLink 3.0 (A100): expected ~300 GB/s bidirectional")

    if "cross_node_2gpu_allreduce" in all_results:
        best = max(all_results["cross_node_2gpu_allreduce"], key=lambda x: x["bus_bw_gbps"])
        print(f"  Cross-node 2-GPU all-reduce: {best['bus_bw_gbps']:>6.1f} GB/s bus BW @ {best['size_mb']} MB")
        if best["bus_bw_gbps"] > 20:
            print(f"    -> EFA or InfiniBand (good)")
        elif best["bus_bw_gbps"] > 5:
            print(f"    -> Moderate (possibly EFA with suboptimal config)")
        else:
            print(f"    -> LOW — likely TCP/Ethernet only (EFA not enabled?)")
            print(f"    -> This explains slow multi-node backward passes")

    if "cross_node_8gpu_allreduce" in all_results:
        best = max(all_results["cross_node_8gpu_allreduce"], key=lambda x: x["bus_bw_gbps"])
        print(f"  Cross-node 8-GPU all-reduce: {best['bus_bw_gbps']:>6.1f} GB/s bus BW @ {best['size_mb']} MB")

    # Save results
    results_path = "/tmp/interconnect_benchmark.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
