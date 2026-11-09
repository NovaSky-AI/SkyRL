"""Head-full DSA all-to-alls (``patch_dsa_head_full.py``) against Megatron's and an explicit reference.

Under torchrun, every rank checks that ``_HeadToSeq`` / ``_SeqToHead``:

- without zigzag, equal ``all_to_all_hp2sp`` / ``all_to_all_sp2hp`` in values and input gradients;
- with zigzag, give rank r the rows of chunk r of the first half and chunk ``TP - 1 - r`` of the
  second half with every rank's columns (checked against an all-gather), invert each other, and
  backpropagate exactly.

Those comparisons are bitwise: the all-to-alls only move data. It also runs the wrapped hook
(``_head_full``) end to end, with a PyTorch reference sparse attention as the backend, against the
same reference on the rank's own heads (the unwrapped TP layout): outputs and query gradients per
rank, and key gradients summed over the TP group (each rank holds a partial sum, over its heads
unwrapped and over its query rows head-full), for contiguous and zigzag rows. Those are float32
and differ only in summation order.
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
    mine = torch.cat([torch.arange(rank * chunk, (rank + 1) * chunk), torch.arange(second, second + chunk)]).cuda()
    expected = torch.cat([g[mine] for g in gathered], dim=1)

    y = _HeadToSeq.apply(x, group, True)
    z = _SeqToHead.apply(y * 2, group, True)
    grad = torch.randn_like(z)
    (z * grad).sum().backward()
    return torch.equal(y, expected) and torch.equal(z, x.detach() * 2) and torch.equal(x.grad, grad * 2)


def _reference_sparse_attention(query, key, topk_indices, softmax_scale, v_channels, topk_length=None):
    """Absorbed-MLA sparse attention in PyTorch, with the backend hook's signature.

    ``query [sq, b, h, d]``, ``key [sk, b, 1, d]`` (one latent head), ``topk_indices [b, sq, topk]``
    (-1 = unused slot) -> ``[sq, b, h, v_channels]``; values are the key's first ``v_channels``.
    """
    assert topk_length is None
    b = query.size(1)
    kv = key[:, :, 0].transpose(0, 1)  # [b, sk, d]
    idx = topk_indices.long()
    rows = kv[torch.arange(b, device=kv.device)[:, None, None], idx.clamp(min=0)]  # [b, sq, topk, d]
    scores = torch.einsum("sbhd,bstd->bsht", query, rows) * softmax_scale
    scores = scores.masked_fill((idx < 0)[:, :, None, :], float("-inf"))
    out = torch.einsum("bsht,bstv->bshv", scores.softmax(-1), rows[..., :v_channels])
    return out.permute(1, 0, 2, 3)


def _check_head_full_attention(group) -> bool:
    """``_head_full`` around the reference vs the reference on this rank's heads, contiguous and zigzag."""
    from skyrl.backends.skyrl_train.patches.megatron import patch_dsa_head_full

    rank, tp = group.rank(), group.size()
    sq, b, heads, d, v, topk = 16 * tp, 2, 4 * tp, 64, 48, 12
    gen = torch.Generator(device="cuda").manual_seed(1234)  # same full tensors on every rank
    query = torch.randn(sq, b, heads, d, device="cuda", generator=gen)
    key = torch.randn(sq, b, 1, d, device="cuda", generator=gen)
    grad_out = torch.randn(sq, b, heads, v, device="cuda", generator=gen)
    # Causal top-k: row s picks up to `topk` distinct keys in [0, s]; unused slots are -1.
    scores = torch.rand(b, sq, sq, device="cuda", generator=gen)
    scores = scores.masked_fill(torch.ones(sq, sq, device="cuda", dtype=torch.bool).triu(1), -1.0)
    values, indices = scores.topk(topk, dim=-1)
    topk_indices = indices.masked_fill(values < 0, -1).to(torch.int32)
    local = slice(rank * heads // tp, (rank + 1) * heads // tp)
    scale = d**-0.5

    def run(fn):
        q = query[:, :, local].clone().requires_grad_()
        k = key.clone().requires_grad_()
        out = fn(q, k, topk_indices, scale, v)
        out.backward(grad_out[:, :, local])
        key_grad = k.grad.clone()
        dist.all_reduce(key_grad, group=group)
        return out.detach(), q.grad, key_grad

    ref_out, ref_dq, ref_dk = run(_reference_sparse_attention)
    ok = True
    for zigzag in (False, True):
        patch_dsa_head_full._ZIGZAG = zigzag
        out, dq, dk = run(patch_dsa_head_full._head_full(_reference_sparse_attention))
        for got, want in ((out, ref_out), (dq, ref_dq), (dk, ref_dk)):
            ok &= torch.allclose(got, want, rtol=1e-5, atol=1e-5)
    patch_dsa_head_full._ZIGZAG = False
    return ok


def _distributed_main() -> None:
    from megatron.core import parallel_state

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=dist.get_world_size())
    group = parallel_state.get_tensor_model_parallel_group()
    ok = _check_matches_megatron(group) and _check_zigzag(group) and _check_head_full_attention(group)
    flag = torch.tensor([1.0 if ok else 0.0], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"RESULT: {'PASS' if flag.item() == 1.0 else 'FAIL'} (TP={dist.get_world_size()})")
    parallel_state.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.megatron
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
