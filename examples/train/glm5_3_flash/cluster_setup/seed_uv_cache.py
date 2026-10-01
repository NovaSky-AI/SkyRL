"""Copy a shared uv cache onto every Ray node's local disk (GPU workers in parallel, then the head).

`uv run --isolated` builds a fresh virtualenv for every process: the job driver on the head node
and each Ray worker. uv fills it by hardlinking from its cache, but it cannot hardlink across
filesystems, so building from a cache on network storage copies the whole cache (~60 GB) per
process. Workers then miss Ray's worker-registration timeout, and drivers stall for 15+ minutes.
A node-local copy of the cache makes every env build take seconds.

    python seed_uv_cache.py [SHARED_CACHE] [LOCAL_CACHE]

Defaults: $SHARED_UV_CACHE (or /shared/hackskyrl/uv_cache) -> ~/.cache/uv on every node.
"""

import os
import socket
import subprocess
import sys
import time

import ray

SHARED = (sys.argv[1] if len(sys.argv) > 1 else os.environ.get("SHARED_UV_CACHE", "/shared/hackskyrl/uv_cache")).rstrip(
    "/"
) + "/"
LOCAL = (sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/.cache/uv")).rstrip("/") + "/"
RSYNC = ["rsync", "-a", "--exclude", "builds-v0/", "--exclude", ".tmp*", SHARED, LOCAL]


def _copy():
    t = time.time()
    os.makedirs(LOCAL, exist_ok=True)
    p = subprocess.run(RSYNC, capture_output=True, text=True)
    du = subprocess.run(["du", "-sh", LOCAL], capture_output=True, text=True).stdout.split()[0]
    return socket.gethostname(), p.returncode, round(time.time() - t), du, p.stderr[-300:]


@ray.remote(num_cpus=4)
def _seed_remote():
    return _copy()


if __name__ == "__main__":
    if not os.path.isdir(SHARED):
        sys.exit(f"shared uv cache {SHARED} does not exist; run `uv sync` with UV_CACHE_DIR={SHARED} first")
    ray.init(address="auto", logging_level="ERROR")
    nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("GPU", 0) > 0]
    print(f"seeding {len(nodes)} GPU nodes from {SHARED} -> {LOCAL}", flush=True)
    refs = [_seed_remote.options(resources={f"node:{n['NodeManagerAddress']}": 0.01}).remote() for n in nodes]
    ok = True
    for host, rc, secs, du, err in ray.get(refs):
        print(f"  {host}: rc={rc} {secs}s {du}", "" if rc == 0 else err, flush=True)
        ok &= rc == 0
    print("seeding the head node (this machine)", flush=True)
    host, rc, secs, du, err = _copy()
    print(f"  {host}: rc={rc} {secs}s {du}", "" if rc == 0 else err, flush=True)
    ok &= rc == 0
    sys.exit(0 if ok else 1)
