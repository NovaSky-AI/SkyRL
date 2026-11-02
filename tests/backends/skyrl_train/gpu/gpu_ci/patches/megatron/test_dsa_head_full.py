"""Head-full DSA all-to-alls (``patch_dsa_head_full.py``) against Megatron's and an explicit reference.

Under torchrun, every rank checks that ``_HeadToSeq`` / ``_SeqToHead``:

- without zigzag, equal ``all_to_all_hp2sp`` / ``all_to_all_sp2hp`` in values and input gradients;
- with zigzag, give rank r the rows of chunk r of the first half and chunk ``TP - 1 - r`` of the
  second half with every rank's columns (checked against an all-gather), invert each other, and
  backpropagate exactly.

All comparisons are bitwise: the all-to-alls only move data.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from skyrl.train.utils.utils import get_free_port


def _check_matches_megatron(group) -> bool:
    from megatron.core.tensor_parallel.mappings import (
        all_to_all_hp2sp,
        all_to_all_sp2hp,
    )

    from skyrl.backends.skyrl_train.patches.megatron.patch_dsa_head_full import (
        _HeadToSeq,
        _SeqToHead,
    )

    ok = True
    for rows, cols in [(4096, 16 * 512), (4 * 1000, 24)]:
        x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_()
        y, y_ref = _HeadToSeq.apply(x, group, False), all_to_all_hp2sp(x_ref, group=group)
        scale = torch.randn_like(y)
        z, z_ref = _SeqToHead.apply(y * scale, group, False), all_to_all_sp2hp(y_ref * scale, group=group)
        grad = torch.randn_like(z)
        (z * grad).sum().backward()
        (z_ref * grad).sum().backward()
        ok &= torch.equal(y, y_ref) and torch.equal(z, z_ref) and torch.equal(x.grad, x_ref.grad)
    return ok


def _check_zigzag(group) -> bool:
    from skyrl.backends.skyrl_train.patches.megatron.patch_dsa_head_full import (
        _HeadToSeq,
        _SeqToHead,
    )

    rank, tp = group.rank(), group.size()
    rows, cols = 8 * tp * 37, 96
    x = torch.randn(rows, cols, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    gathered = [torch.empty_like(x) for _ in range(tp)]
    dist.all_gather(gathered, x.detach(), group=group)
    chunk = rows // (2 * tp)
    second = rows // 2 + (tp - 1 - rank) * chunk
    mine = torch.cat(
        [torch.arange(rank * chunk, (rank + 1) * chunk), torch.arange(second, second + chunk)]
    ).cuda()
    expected = torch.cat([g[mine] for g in gathered], dim=1)

    y = _HeadToSeq.apply(x, group, True)
    z = _SeqToHead.apply(y * 2, group, True)
    grad = torch.randn_like(z)
    (z * grad).sum().backward()
    return torch.equal(y, expected) and torch.equal(z, x.detach() * 2) and torch.equal(x.grad, grad * 2)


def _distributed_main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    group = dist.new_group(list(range(dist.get_world_size())))
    ok = _check_matches_megatron(group) and _check_zigzag(group)
    flag = torch.tensor([1.0 if ok else 0.0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"RESULT: {'PASS' if flag.item() == 1.0 else 'FAIL'} (TP={dist.get_world_size()})")
    dist.destroy_process_group()


@pytest.mark.parametrize("nproc", [2, 4])
def test_head_full_all_to_alls(nproc):
    """Spawn torchrun --nproc_per_node=nproc and check the head-full all-to-alls on every rank."""
    if torch.cuda.device_count() < nproc:
        pytest.skip(f"needs >= {nproc} GPUs, have {torch.cuda.device_count()}")
    env = dict(os.environ, MASTER_ADDR="localhost", MASTER_PORT=str(get_free_port()))
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_port",
        env["MASTER_PORT"],
        __file__,
    ]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert "RESULT: PASS" in (res.stdout + res.stderr), f"TP={nproc} failed:\n{res.stdout}\n{res.stderr}"


if __name__ == "__main__":
    _distributed_main()
