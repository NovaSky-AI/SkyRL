"""
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_chunked_logprob_backward.py
"""

import os

import pytest
import torch
import torch.distributed as dist

from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
    ChunkedDistributedLogprob,
    DistributedLogprob,
)
from skyrl.train.utils.utils import get_free_port


@pytest.fixture(scope="module")
def tp_group():
    """Single-rank TP process group used by both autograd functions.

    Uses gloo distributed backend for simplicity because the world size is 1.
    """
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield dist.group.WORLD
    if dist.is_initialized():
        dist.destroy_process_group()


def _forward_backward(func_cls, logits, target, vocab_start, vocab_end, tp_group, *, chunk_size=None):
    """Run forward+backward through a logprob autograd function and return (out, grad_logits).

    Uses a non-uniform upstream gradient so that any per-position bug surfaces.
    """
    leaf = logits.detach().clone().requires_grad_(True)
    if chunk_size is None:
        out = func_cls.apply(leaf, target, vocab_start, vocab_end, tp_group, False)
    else:
        out = func_cls.apply(leaf, target, vocab_start, vocab_end, chunk_size, tp_group, False)
    grad_seed = torch.linspace(0.5, 1.5, steps=out.numel(), device=out.device, dtype=out.dtype).reshape(out.shape)
    out.backward(grad_seed)
    return out.detach(), leaf.grad.detach()


@pytest.mark.parametrize("chunk_size", [1, 7, 16, 64, 512])
@pytest.mark.parametrize("with_oov_targets", [False, True])
def test_chunked_matches_non_chunked(tp_group, chunk_size, with_oov_targets):
    """Chunked and non-chunked logprob produce matching forwards and gradients.

    Sweeps several chunk sizes (including one larger than the sequence length,
    which collapses the chunk loop to a single iteration) and toggles whether
    targets can fall outside the TP rank's vocab slice. The out-of-vocab path
    exercises the ``target_mask`` branch in both functions.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)

    batch_size = 4
    seq_len = 32
    vocab_size = 32_000

    target_high = vocab_size + 1024 if with_oov_targets else vocab_size

    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.bfloat16, device=device) * 2.0
    target = torch.randint(0, target_high, (batch_size, seq_len), device=device, dtype=torch.long)

    out_ref, grad_ref = _forward_backward(DistributedLogprob, logits, target, 0, vocab_size, tp_group)
    out_chunk, grad_chunk = _forward_backward(
        ChunkedDistributedLogprob,
        logits,
        target,
        0,
        vocab_size,
        tp_group,
        chunk_size=chunk_size,
    )

    # Output dtype contract: log-softmax upcasts to fp32 internally.
    assert out_chunk.dtype == torch.float32

    # Forward parity. Both paths do the same fp32 math, but for small chunk sizes
    # the reduction order across chunks differs from the single-shot path, which
    # introduces ~1e-6 rounding noise. A loose tolerance still rules out real bugs.
    torch.testing.assert_close(out_chunk, out_ref, atol=1e-5, rtol=1e-5)

    # Gradient parity. Both paths use the same scatter-add formulation, so the
    # tolerance can be tight relative to the bf16-logits/fp32-grad pipeline.
    torch.testing.assert_close(grad_chunk, grad_ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize(
    "case",
    [
        # (batch, seq_len, vocab, chunk_size, mask_mode)
        # mask_mode: "default" (mixed), "all_in" (no OOV), "all_out" (all OOV)
        pytest.param((1, 1, 1024, 4, "default"), id="seq1"),
        pytest.param((2, 8, 1024, 32, "all_in"), id="all_in_vocab"),
        pytest.param((2, 8, 1024, 32, "all_out"), id="all_out_vocab"),
        pytest.param((2, 8, 8, 4, "default"), id="tiny_vocab"),
    ],
)
def test_chunked_matches_non_chunked_edge_cases(tp_group, case):
    """Edge cases for the chunked path: short sequences, mask extremes, tiny vocab.

    Covers configurations the main sweep does not: ``seq_len=1`` (chunk loop runs
    once with ``chunk_len=1``), a target_mask that is entirely False or entirely
    True (the empty-``scatter_add_`` path), and a very small vocab that stresses
    the masked-select/scatter arithmetic.
    """
    batch_size, seq_len, vocab_size, chunk_size, mask_mode = case
    device = torch.device("cuda")
    torch.manual_seed(1)

    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.bfloat16, device=device) * 2.0
    if mask_mode == "all_out":
        # Every target is outside the TP rank's vocab slice [0, vocab_size).
        target = torch.full((batch_size, seq_len), vocab_size + 5, device=device, dtype=torch.long)
    else:
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    out_ref, grad_ref = _forward_backward(DistributedLogprob, logits, target, 0, vocab_size, tp_group)
    out_chunk, grad_chunk = _forward_backward(
        ChunkedDistributedLogprob,
        logits,
        target,
        0,
        vocab_size,
        tp_group,
        chunk_size=chunk_size,
    )

    torch.testing.assert_close(out_chunk, out_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(grad_chunk, grad_ref, atol=1e-5, rtol=1e-4)


def test_chunked_backward_uses_scatter_add_path(tp_group):
    """Chunked backward uses ``scatter_add_`` and does not materialize one_hot.

    Asserts the implementation contract directly: ``scatter_add_`` must be
    invoked during backward, and ``torch.nn.functional.one_hot`` must not.
    Verifying the code path via mocks is more reliable than a peak-memory
    assertion, which is sensitive to caching allocator behaviour and other
    in-flight tensors.

    We patch ``F.one_hot`` after ``forward`` has finished (forward doesn't use
    it either; restricting the patch window keeps the assertion tight to the
    backward path).
    """
    from unittest.mock import patch

    device = torch.device("cuda")
    torch.manual_seed(0)

    # chunk_size >= seq_len collapses the chunk loop to a single iteration --
    # the path most likely to regress to a one_hot formulation.
    batch_size = 2
    seq_len = 16
    vocab_size = 1024
    chunk_size = 1024

    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.bfloat16, device=device, requires_grad=True)
    target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    out = ChunkedDistributedLogprob.apply(logits, target, 0, vocab_size, chunk_size, tp_group, False)

    real_scatter_add_ = torch.Tensor.scatter_add_
    scatter_add_calls = []

    def _tracking_scatter_add_(self, dim, index, src):
        scatter_add_calls.append((tuple(self.shape), int(dim), tuple(index.shape)))
        return real_scatter_add_(self, dim, index, src)

    with (
        patch(
            "torch.nn.functional.one_hot",
            side_effect=AssertionError("one_hot must not be called in chunked backward"),
        ),
        patch.object(torch.Tensor, "scatter_add_", _tracking_scatter_add_),
    ):
        out.sum().backward()

    assert scatter_add_calls, "Chunked backward must call scatter_add_ to place chosen-token grads"
