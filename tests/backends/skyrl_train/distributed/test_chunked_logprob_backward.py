"""Gradient-parity tests for ``ChunkedDistributedLogprob`` vs ``DistributedLogprob``.

These tests verify that the chunked log-probability autograd function produces
the same forward and backward results as the non-chunked baseline for a variety
of chunk sizes, including the OOM-regression path (chunk_size >= seq_len) that
caused the GSM8K bug described in the PR.

The tests use TP=1 (single GPU) and only exercise the gradient/forward math --
no distributed setup is required because both functions handle the single-rank
case via the WORLD group.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

_REQUIRES_CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _load_model_utils():
    """Load model_utils.py without importing the whole skyrl package.

    The top-level ``skyrl`` package imports modules that trigger Pydantic schema
    generation requiring Python 3.12+. ``model_utils.py`` itself has no
    project-internal imports, so we load it directly from disk to keep this test
    runnable on the project's supported Python versions without dragging in the
    rest of the package.
    """
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "skyrl" / "backends" / "skyrl_train" / "distributed" / "megatron" / "model_utils.py"
    spec = importlib.util.spec_from_file_location("_skyrl_model_utils_under_test", module_path)
    assert spec is not None and spec.loader is not None, f"could not load {module_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _ensure_single_rank_process_group() -> torch.distributed.ProcessGroup:
    """Init a single-rank gloo+nccl process group if one isn't already set up."""
    if not dist.is_available():
        pytest.skip("torch.distributed not available")
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29555")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend, rank=0, world_size=1)
    return dist.group.WORLD


def _run_logprob(
    func_cls,
    logits: torch.Tensor,
    target: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
    tp_group,
    *,
    chunk_size: int | None = None,
):
    """Forward+backward through one of the two logprob autograd functions.

    Returns ``(logprobs, grad_logits)``.
    """
    leaf = logits.detach().clone().requires_grad_(True)
    if chunk_size is None:
        out = func_cls.apply(leaf, target, vocab_start, vocab_end, tp_group, False)
    else:
        out = func_cls.apply(leaf, target, vocab_start, vocab_end, chunk_size, tp_group, False)
    # Use a non-uniform upstream grad so any per-position bug surfaces.
    grad_seed = torch.linspace(0.5, 1.5, steps=out.numel(), device=out.device, dtype=out.dtype).reshape(out.shape)
    out.backward(grad_seed)
    return out.detach(), leaf.grad.detach()


@_REQUIRES_CUDA
@pytest.mark.parametrize("chunk_size", [1, 7, 16, 64, 512])
@pytest.mark.parametrize("with_oov_targets", [False, True])
def test_chunked_matches_non_chunked(chunk_size: int, with_oov_targets: bool):
    """ChunkedDistributedLogprob and DistributedLogprob must produce matching grads.

    We test several chunk sizes -- including ``chunk_size > seq_len`` (the OOM
    regression path) -- and also a configuration where some targets fall outside
    the TP rank's vocab slice so that the ``target_mask`` branch is exercised.
    """
    mod = _load_model_utils()
    ChunkedDistributedLogprob = mod.ChunkedDistributedLogprob
    DistributedLogprob = mod.DistributedLogprob

    tp_group = _ensure_single_rank_process_group()

    device = torch.device("cuda")
    torch.manual_seed(0)

    batch_size = 4
    seq_len = 32
    vocab_size = 32_000

    # Targets drawn from a possibly larger range to exercise the OOV / target_mask path.
    target_high = vocab_size + 1024 if with_oov_targets else vocab_size

    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.bfloat16, device=device) * 2.0
    target = torch.randint(0, target_high, (batch_size, seq_len), device=device, dtype=torch.long)

    vocab_start = 0
    vocab_end = vocab_size

    out_ref, grad_ref = _run_logprob(
        DistributedLogprob,
        logits,
        target,
        vocab_start,
        vocab_end,
        tp_group,
    )
    out_chunk, grad_chunk = _run_logprob(
        ChunkedDistributedLogprob,
        logits,
        target,
        vocab_start,
        vocab_end,
        tp_group,
        chunk_size=chunk_size,
    )

    # Forward parity: both paths do the same fp32 math, but for small chunk
    # sizes the reduction order over chunks can differ from the single-shot
    # path, leading to tiny (~1e-6) rounding noise. A loose tolerance still
    # rules out any real bug.
    torch.testing.assert_close(out_chunk, out_ref, atol=1e-5, rtol=1e-5)

    # Gradient parity. The fast path performs exactly the same ops (neg / mul_ /
    # scatter_add_) as DistributedLogprob, just per-chunk, so we use a tight
    # tolerance suited to the bf16 logits / fp32 grad pipeline.
    torch.testing.assert_close(grad_chunk, grad_ref, atol=1e-5, rtol=1e-4)


@_REQUIRES_CUDA
def test_chunked_backward_does_not_call_one_hot():
    """Regression: backward must not materialize ``one_hot(masked_target, V)``.

    The pre-fix ``ChunkedDistributedLogprob.backward`` allocated an int64 one-hot
    tensor of shape ``[B, chunk_len, partition_vocab_size]`` per chunk. For the
    Qwen3-0.6B GSM8K config (V=151936, B=128, S~280, chunk_size=1024) this is
    ~43 GB of int64 plus another ~22 GB float copy -- causing the OOM the user
    reported.

    The fast path uses ``scatter_add_`` on a flat view of the existing
    ``softmax`` buffer, so ``torch.nn.functional.one_hot`` should never be
    called during backward. This test verifies that directly via mock-patching,
    which is far more reliable than a memory-bound check (peak memory depends
    on caching allocator behaviour, other in-flight fp32 tensors, etc.).
    """
    from unittest.mock import patch

    mod = _load_model_utils()
    ChunkedDistributedLogprob = mod.ChunkedDistributedLogprob

    tp_group = _ensure_single_rank_process_group()
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Use chunk_size > seq_len so the whole sequence is one chunk -- this is the
    # exact regime that hit the OOM in the bug report.
    batch_size = 2
    seq_len = 16
    vocab_size = 1024
    chunk_size = 1024

    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.bfloat16, device=device, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    out = ChunkedDistributedLogprob.apply(logits, target, 0, vocab_size, chunk_size, tp_group, False)

    # Patch one_hot to raise if called during backward. We patch on the
    # torch.nn.functional module after forward (forward doesn't use one_hot
    # either, but we keep the patch tight to backward to avoid false positives
    # from other code paths).
    with patch("torch.nn.functional.one_hot", side_effect=AssertionError("one_hot must not be called in backward")):
        out.sum().backward()
