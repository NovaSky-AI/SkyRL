"""Utils methods for checking. Caller uses `run_nccl_torch_distributed_check()` to determine if NCCL P2P/SHM is supported."""

import os
import torch
import socket
import torch.distributed as dist
import torch.multiprocessing as mp
import time
from datetime import timedelta

def _nccl_reduce_worker(rank: int, world_size: int, master_addr: str, master_port: int, env_overrides: dict):
    for k, v in (env_overrides or {}).items():
        os.environ[str(k)] = str(v)

    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    torch.cuda.set_device(rank)
    # Short process-group timeout to fail fast inside worker if possible
    try:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=10),
        )
    except Exception:
        # Ensure non-zero exit on failure to init
        raise

    try:
        tensor = torch.ones(1, device=torch.device(f"cuda:{rank}"))
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        if abs(float(tensor.item()) - float(world_size)) > 1e-3:
            raise RuntimeError(f"Unexpected all_reduce result: {tensor.item()} vs {world_size}")
    finally:
        try:
            dist.destroy_process_group()
        except Exception:
            pass


def run_nccl_torch_distributed_check(env_overrides: dict | None, timeout_s: int = 15) -> bool:
    world_size = torch.cuda.device_count()
    assert world_size >= 2

    # Allocate a free TCP port for init_method
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    ctx = mp.get_context("spawn")
    procs: list[mp.Process] = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_nccl_reduce_worker,
            args=(rank, world_size, "127.0.0.1", port, env_overrides or {}),
        )
        p.daemon = True
        procs.append(p)

    start = time.time()
    for p in procs:
        p.start()

    success = True
    while time.time() - start < timeout_s:
        if all(p.exitcode is not None for p in procs):
            break
        time.sleep(0.1)

    # If any are still alive after timeout, terminate and mark failure
    for p in procs:
        if p.exitcode is None:
            try:
                p.terminate()
            except Exception:
                pass
            success = False

    # Check exit codes
    for p in procs:
        p.join(timeout=1)
        if p.exitcode != 0:
            success = False

    return success
