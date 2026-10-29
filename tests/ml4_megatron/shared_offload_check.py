"""Data-parallel-replicated frozen weights share one node-local offload file; anything else stays private.

SKYRL_FROZEN_OFFLOAD_DIR=<dir> SKYRL_FROZEN_OFFLOAD_TOKEN=<t> torchrun --nproc-per-node 4 \
    tests/ml4_megatron/shared_offload_check.py

Every rank offloads: two dense weights equal on all ranks, one dense weight that differs on the last rank,
and one expert weight (differs per rank). Checks, per param, which file inode backs each rank's mapping,
that the restored tensors equal the originals bitwise, and that no shared file is left on disk.
Exits non-zero on any failure.
"""

import os
import sys

import torch
import torch.distributed as dist


def mapped_inode(t):
    lo = t.data_ptr()
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.split()
            a, b = (int(x, 16) for x in parts[0].split("-"))
            if a <= lo < b:
                return int(parts[4]), " ".join(parts[5:])
    return None, None


def main() -> int:
    from skyrl.backends.skyrl_train.distributed.megatron import megatron_utils as mu

    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    g = torch.Generator(device="cuda").manual_seed(0)
    dense_a = torch.randn(4096, 3000, device="cuda", generator=g, dtype=torch.bfloat16)
    dense_b = torch.randn(1000, 77, device="cuda", generator=g, dtype=torch.bfloat16)
    odd = torch.randn(512, 512, device="cuda", generator=g, dtype=torch.bfloat16)
    if rank == world - 1:
        odd[7, 7] += 1
    expert = torch.full((256, 256), float(rank), device="cuda", dtype=torch.bfloat16)
    params = {
        "decoder.layers.0.self_attention.linear_qkv.weight": dense_a,
        "embedding.word_embeddings.weight": dense_b,
        "decoder.layers.1.mlp.linear_fc1.weight": odd,
        "decoder.layers.0.mlp.experts.linear_fc1.weight0": expert,
    }
    want = {n: t.clone() for n, t in params.items()}
    ps = {}
    for n, t in params.items():
        p = torch.nn.Parameter(t, requires_grad=False)
        if ".experts." in n:
            p.allreduce = False
        assert mu._offload_frozen_param_to_file(n, p), n
        p.data = torch.empty(0, dtype=t.dtype, device=t.device)
        ps[n] = p
    mu._unlink_shared_offload_files()

    rows = {n: mapped_inode(p._offload_cpu_data) for n, p in ps.items()}
    restored = {n: bool(torch.equal(p._offload_cpu_data.to("cuda"), want[n])) for n, p in ps.items()}
    token_dir = os.path.join(mu._FROZEN_OFFLOAD_DIR, f"shared-{os.environ['SKYRL_FROZEN_OFFLOAD_TOKEN']}")
    left = sorted(os.listdir(token_dir))
    gathered = [None] * world
    dist.all_gather_object(gathered, (rows, restored, left))
    ok = True
    if rank == 0:
        inodes = {n: [g[0][n][0] for g in gathered] for n in params}
        paths = {n: [g[0][n][1] for g in gathered] for n in params}
        for n in params:
            print(n, "inodes", inodes[n], "paths", paths[n], flush=True)
        expect_shared = {n: len(set(inodes[n])) == 1 for n in list(params)[:2]}
        odd_n, exp_n = list(params)[2], list(params)[3]
        checks = {
            "dense shared on all ranks": all(expect_shared.values())
            and all("shared-" in p for n in list(params)[:2] for p in paths[n]),
            "mismatching rank fell back to private": "shared-" not in paths[odd_n][-1]
            and len(set(inodes[odd_n][:-1])) == 1,
            "expert weights private": all("shared-" not in p for p in paths[exp_n])
            and len(set(inodes[exp_n])) == world,
            "restored bitwise": all(all(g[1].values()) for g in gathered),
            "no shared files left on disk": all(not g[2] for g in gathered),
        }
        for k, v in checks.items():
            print(f"{k}: {v}", flush=True)
        ok = all(checks.values())
        print("PASS" if ok else "FAIL", flush=True)
    flag = [ok]
    dist.broadcast_object_list(flag, src=0)
    dist.destroy_process_group()
    return 0 if flag[0] else 1


if __name__ == "__main__":
    sys.exit(main())
