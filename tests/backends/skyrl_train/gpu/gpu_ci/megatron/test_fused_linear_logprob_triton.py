"""GPU correctness tests for the Triton fused LM-head log-prob backend.

Exercises the VENDORED Triton kernel adapter ``FusedLinearLogprobTriton`` on CUDA and asserts it
matches the STOCK materialized-logits path (``ChunkedDistributedLogprob`` / ``DistributedLogprob``
over ``hidden @ W_shard.T``) for:
  * forward log-prob (full-vocab, post TP all-reduce),
  * grad w.r.t. hidden (this rank's PARTIAL — neither side all-reduces it), and
  * grad w.r.t. weight (per-shard).
across TP=1 and TP=2, with and without out-of-vocab targets, multiple chunk sizes, and BOTH
bf16 and fp32 inputs.

This is the ``fused_backend="triton"`` selector's kernel-level correctness check. The reconciled
integration (on top of upstream #1841) dispatches to this same kernel from
``model_utils.from_parallel_hidden_to_logprobs(..., fused_backend="triton")``; here we drive the
autograd Function directly so the test is independent of the wrapper/dispatch plumbing.

Precision: the fp32 parametrizations flip the kernel into bitwise-faithful IEEE fp32 (via
``FORCE_FP32_IEEE_PRECISION = True``) and assert the kernel math is EXACT at a tight 1e-4
tolerance — a committed, reproducible "the kernel is mathematically correct" sanity check. The
bf16 parametrizations run at PRODUCTION precision (the flag stays False => the fast TF32 Hopper
path that ``apply()`` uses in production) and check the looser ~2e-2 bf16 tolerance. The flag is
set/reset inside each spawned worker and is reliably reset in a finally block so it never leaks.

Requires a CUDA device AND triton (the kernel is GPU-only); skipped otherwise.

    uv run --isolated --extra dev --extra megatron -- pytest -s \
        tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_fused_linear_logprob_triton.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from skyrl.backends.skyrl_train.distributed.megatron import fused_linear_logprob_triton
from skyrl.backends.skyrl_train.distributed.megatron.fused_linear_logprob_triton import (
    TRITON_AVAILABLE,
    FusedLinearLogprobTriton,
)

# Module-level skip: the whole file needs a GPU and triton.
pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and TRITON_AVAILABLE),
    reason="Triton fused LM-head log-prob requires a CUDA device and triton",
)


def _direct_fused_logprobs(hidden, weight_shard, target_shifted, vstart, vend, chunk_size, grad_seed):
    """Drive FusedLinearLogprobTriton.apply directly (bypasses model_utils dispatch).

    The public entry point shifts targets internally; here we pass an ALREADY-shifted target and
    compare against a directly-driven stock reference, so this path is independent of whether
    model_utils has been wired to select the Triton backend yet.
    """
    leaf_h = hidden.detach().clone().requires_grad_(True)
    leaf_w = weight_shard.detach().clone().requires_grad_(True)
    lp = FusedLinearLogprobTriton.apply(
        leaf_h, leaf_w, target_shifted, vstart, vend, chunk_size, dist.group.WORLD, False
    )
    lp.backward(grad_seed.clone())
    return lp.detach(), leaf_h.grad.detach(), leaf_w.grad.detach()


def _stock_shifted(hidden, weight_shard, target_shifted, vstart, vend, chunk_size):
    """Stock logprob over materialized logits, given ALREADY-shifted targets.

    Replicates from_parallel_logits_to_logprobs' DistributedLogprob/ChunkedDistributedLogprob math
    on pre-shifted targets (no internal roll), to compare against _direct_fused_logprobs."""
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        ChunkedDistributedLogprob,
        DistributedLogprob,
    )

    leaf_h = hidden.detach().clone().requires_grad_(True)
    leaf_w = weight_shard.detach().clone().requires_grad_(True)
    logits = leaf_h @ leaf_w.t()
    seq_len = logits.shape[1]
    if chunk_size is not None and chunk_size < seq_len:
        lp = ChunkedDistributedLogprob.apply(logits, target_shifted, vstart, vend, chunk_size, dist.group.WORLD, False)
    else:
        lp = DistributedLogprob.apply(logits, target_shifted, vstart, vend, dist.group.WORLD, False)
    grad_seed = torch.linspace(0.5, 1.5, steps=lp.numel(), device=lp.device, dtype=lp.dtype).reshape(lp.shape)
    lp.backward(grad_seed)
    return lp.detach(), leaf_h.grad.detach(), leaf_w.grad.detach(), grad_seed


def _tol_for_dtype(dtype):
    # bf16 inputs: ~2e-2 (production-precision path); fp32 inputs: tight ~1e-4 (IEEE sanity check).
    if dtype == torch.bfloat16:
        return dict(atol=2e-2, rtol=2e-2)
    return dict(atol=1e-4, rtol=1e-4)


def _worker(rank, world_size, port, chunk_size, with_oov, dtype_str, ret_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # gloo is fine for the tiny TP all-reduces (verl's kernel itself runs on CUDA tensors; only
    # the all-reduce of label-logit / softmax-stats crosses ranks, and those are small).
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # PRECISION CONTRACT (deliberate, see fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION):
    #   * fp32 parametrizations flip the kernel into bitwise-faithful IEEE fp32 so we can assert the
    #     kernel MATH IS EXACT against the IEEE-fp32 PyTorch reference at the tight 1e-4 tolerance.
    #     This is the committed, reproducible "kernel is mathematically correct" sanity check.
    #   * bf16 parametrizations run the kernel at PRODUCTION precision (flag False => TF32 fast path),
    #     i.e. exactly what apply() does in production, checked at the looser ~2e-2 bf16 tolerance.
    # We set/reset the module-level flag here (inside the spawned worker) because mp spawn starts a
    # fresh interpreter — a parent-process monkeypatch would not propagate to the child. The
    # try/finally guarantees we never leak True out of an fp32 case.
    _prev_force_ieee = fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION
    fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION = dtype_str == "fp32"
    try:
        torch.cuda.set_device(0)
        device = torch.device("cuda")
        dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float32
        torch.manual_seed(0)  # identical across ranks => identical hidden/weight/target

        # hidden_size must be a multiple of 128 for verl's kernel (assert hidden_size % 128 == 0).
        batch_size, seq_len, hidden_size, vocab_size = 3, 24, 128, 256
        hidden = (torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.5).to(dtype)
        weight_full = (torch.randn(vocab_size, hidden_size, device=device) * 0.1).to(dtype)
        target_high = vocab_size + 50 if with_oov else vocab_size
        target = torch.randint(0, target_high, (batch_size, seq_len), device=device, dtype=torch.long)

        assert vocab_size % world_size == 0
        shard = vocab_size // world_size
        vstart, vend = rank * shard, (rank + 1) * shard
        weight_shard = weight_full[vstart:vend].contiguous()

        # Compare on ALREADY-shifted targets to be independent of model_utils dispatch wiring.
        target_shifted = target.roll(shifts=-1, dims=-1)

        lp_ref, gh_ref, gw_ref, grad_seed = _stock_shifted(
            hidden, weight_shard, target_shifted, vstart, vend, chunk_size
        )
        lp_fused, gh_fused, gw_fused = _direct_fused_logprobs(
            hidden, weight_shard, target_shifted, vstart, vend, chunk_size, grad_seed
        )

        tol = _tol_for_dtype(dtype)
        fwd_ok = torch.allclose(lp_fused.float(), lp_ref.float(), **tol)
        gh_ok = torch.allclose(gh_fused.float(), gh_ref.float(), **tol)
        gw_ok = torch.allclose(gw_fused.float(), gw_ref.float(), **tol)

        ret_dict[rank] = {
            "fwd_ok": bool(fwd_ok),
            "gh_ok": bool(gh_ok),
            "gw_ok": bool(gw_ok),
            "lp_dtype": str(lp_fused.dtype),
            "fwd_max_abs": float((lp_fused.float() - lp_ref.float()).abs().max()),
            "gh_max_abs": float((gh_fused.float() - gh_ref.float()).abs().max()),
            "gw_max_abs": float((gw_fused.float() - gw_ref.float()).abs().max()),
        }
    finally:
        # Reliably reset the precision flag so a True never leaks into a later (e.g. bf16) run that
        # happens to reuse this interpreter; then tear down the process group.
        fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION = _prev_force_ieee
        dist.destroy_process_group()


def _run(world_size, chunk_size, with_oov, dtype_str):
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    ret = manager.dict()
    import socket

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]
    mp.spawn(
        _worker,
        args=(world_size, port, chunk_size, with_oov, dtype_str, ret),
        nprocs=world_size,
        join=True,
    )
    return dict(ret)


@pytest.mark.parametrize("dtype_str", ["fp32", "bf16"])
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("chunk_size", [8, 1000])  # 1000 > seq_len => single-chunk path
@pytest.mark.parametrize("with_oov", [False, True])
def test_fused_triton_matches_stock_logits_path(dtype_str, world_size, chunk_size, with_oov):
    """Triton fused hidden->logprob matches the stock materialized-logits path (fwd + both grads).

    world_size=1 checks the kernel math in isolation; world_size=2 checks the vocab-parallel
    online-softmax all-reduce (label-logit + softmax denom), the out-of-shard / fully-OOV target
    masking, and that the per-shard hidden/weight grads match the stock shard reference. fp32 is
    the tight-tolerance IEEE sanity check (kernel flipped to FORCE_FP32_IEEE_PRECISION=True);
    bf16 exercises the PRODUCTION default (fast TF32) precision path at the looser bf16 tolerance.
    """
    if world_size > 1 and torch.cuda.device_count() < world_size:
        pytest.skip(
            f"need >= {world_size} CUDA devices for TP={world_size}; only "
            f"{torch.cuda.device_count()} present (all ranks share device 0 via gloo, "
            f"but multi-process CUDA contexts on one GPU can be flaky)"
        )
    results = _run(world_size, chunk_size, with_oov, dtype_str)
    assert len(results) == world_size
    for rank, r in results.items():
        # The adapter forces fp32 log-probs regardless of input dtype (matches the pure-torch contract).
        assert r["lp_dtype"] == "torch.float32", r
        assert r["fwd_ok"], f"forward mismatch rank={rank}: {r}"
        assert r["gh_ok"], f"grad-hidden mismatch rank={rank}: {r}"
        assert r["gw_ok"], f"grad-weight mismatch rank={rank}: {r}"
