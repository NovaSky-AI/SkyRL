"""CPU parity test for the streamed-buffer ChunkedDistributedLogprob.backward.

``ChunkedDistributedLogprob.backward`` was refactored to stream each chunk's
gradient straight into a single preallocated ``[batch_size, seq_size,
partition_vocab_size]`` fp32 buffer instead of building a Python list of
per-chunk grads and ``torch.cat``-ing it (which doubled peak memory at the cat
moment). The refactor must be byte-identical: the per-chunk log-softmax /
scatter-add math is unchanged and purely per-position (no cross-sequence-position
reduction), so the streamed grad must equal the single-shot ``DistributedLogprob``
grad bit-for-bit at every position, for every chunk size.

This runs on CPU (gloo, world_size=1) so it gates the cheap CI lane; the real
distributed (TP>1) parity lives in
``tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_chunked_logprob_backward_tp.py``.

Run with:
  uv run --isolated --extra skyrl-train --extra dev -- pytest -s \
    tests/backends/skyrl_train/distributed/test_chunked_logprob_backward_streaming.py
"""

import os
import sys
from types import ModuleType

import pytest
import torch
import torch.distributed as dist

from skyrl.backends.skyrl_train.distributed.utils import get_free_port

# ---------------------------------------------------------------------------
# Inject a lightweight megatron stub into sys.modules so that
# ``skyrl...megatron.model_utils`` can be imported without megatron-core
# installed (CPU CI). The autograd functions under test only use torch and
# torch.distributed; the module-level ``import megatron.core.parallel_state``
# is the sole megatron dependency and is never exercised by these tests.
# The stubs are installed/removed by the module-scoped autouse fixture below
# so they do not leak into other test files in the same pytest session,
# matching the convention in test_preprocess_packed_seqs_cp.py.
# ---------------------------------------------------------------------------

_MEGATRON_MODULES = [
    "megatron",
    "megatron.core",
    "megatron.core.parallel_state",
]

_mock_modules: dict[str, ModuleType] = {}
for _name in _MEGATRON_MODULES:
    _mock_modules[_name] = ModuleType(_name)

_mock_modules["megatron.core"].parallel_state = _mock_modules["megatron.core.parallel_state"]


@pytest.fixture(scope="module", autouse=True)
def _stub_megatron_modules():
    """Install the mock ``megatron`` modules for this module's tests only.

    The stubs are injected into ``sys.modules`` at module setup and restored at
    teardown so they do not leak into other test files in the same pytest
    session. ``model_utils``'s only megatron dependency is the module-level
    ``import megatron.core.parallel_state`` (``mpu`` is used only inside
    functions, never at import time), so the stub need only be present when the
    import in each test first runs. We save the prior entries (so a real
    megatron-core on the GPU lane is left untouched) and pop/restore on teardown.
    """
    saved = {_name: sys.modules.get(_name) for _name in _MEGATRON_MODULES}
    for _name in _MEGATRON_MODULES:
        sys.modules[_name] = _mock_modules[_name]
    try:
        yield
    finally:
        for _name in _MEGATRON_MODULES:
            if saved[_name] is None:
                sys.modules.pop(_name, None)
            else:
                sys.modules[_name] = saved[_name]


@pytest.fixture(scope="module")
def tp_group():
    """Single-rank TP process group shared by both autograd functions.

    Uses the gloo backend because the world size is 1, so every ``all_reduce``
    inside ``_compute_distributed_log_softmax`` is the identity. This isolates
    the streamed-buffer refactor from the (separately tested) TP reduction.
    """
    initialized_here = False
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        initialized_here = True
    yield dist.group.WORLD
    if initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def _backward_grad(func_cls, logits, target, vocab_start, vocab_end, tp_group, *, chunk_size=None):
    """Run forward+backward through a logprob autograd function and return the input grad.

    Uses a non-uniform upstream gradient so any per-position bug surfaces.
    """
    leaf = logits.detach().clone().requires_grad_(True)
    if chunk_size is None:
        out = func_cls.apply(leaf, target, vocab_start, vocab_end, tp_group, False)
    else:
        out = func_cls.apply(leaf, target, vocab_start, vocab_end, chunk_size, tp_group, False)
    grad_seed = torch.linspace(0.5, 1.5, steps=out.numel(), device=out.device, dtype=out.dtype).reshape(out.shape)
    out.backward(grad_seed)
    return leaf.grad.detach()


@pytest.mark.parametrize("chunk_size", [1, 3, 7, 16, 32, 64])
@pytest.mark.parametrize("with_oov_targets", [False, True])
def test_streamed_backward_bit_identical_to_non_chunked(tp_group, chunk_size, with_oov_targets):
    """Streamed-buffer chunked grad equals the single-shot grad bit-for-bit.

    The log-softmax + scatter-add backward math is purely per-position, so
    splitting the sequence into chunks (and streaming each chunk into its slice
    of one preallocated buffer) must not change any value. ``torch.equal`` is
    the strongest statement of byte-identity. Sweeps several chunk sizes,
    including ``chunk_size >= seq_len`` (single-iteration / single-write path)
    and chunk sizes that do not evenly divide ``seq_len`` (the ragged last
    chunk, whose ``chunk_end`` clamps to ``seq_len``). ``with_oov_targets``
    exercises the ``target_mask`` (out-of-vocab) branch in both functions.
    """
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        ChunkedDistributedLogprob,
        DistributedLogprob,
    )

    device = torch.device("cpu")
    torch.manual_seed(0)

    batch_size = 4
    # 30 is not a multiple of several of the chunk sizes, so the tiling produces
    # a ragged final chunk; combined with chunk_size=64 > seq_len it also covers
    # the single-chunk path.
    seq_len = 30
    vocab_size = 256

    target_high = vocab_size + 64 if with_oov_targets else vocab_size

    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32, device=device) * 2.0
    target = torch.randint(0, target_high, (batch_size, seq_len), device=device, dtype=torch.long)

    grad_ref = _backward_grad(DistributedLogprob, logits, target, 0, vocab_size, tp_group)
    grad_chunk = _backward_grad(
        ChunkedDistributedLogprob,
        logits,
        target,
        0,
        vocab_size,
        tp_group,
        chunk_size=chunk_size,
    )

    assert grad_chunk.shape == grad_ref.shape == logits.shape
    assert grad_chunk.dtype == torch.float32
    # Byte-identical: per-position math is independent of chunk boundaries.
    assert torch.equal(grad_chunk, grad_ref), "streamed chunked grad must be bit-identical to non-chunked grad"


@pytest.mark.parametrize(
    "case",
    [
        # (batch, seq_len, vocab, chunk_size, mask_mode)
        # mask_mode: "default" (mixed), "all_in" (no OOV), "all_out" (all OOV)
        pytest.param((1, 1, 64, 4, "default"), id="seq1"),
        pytest.param((2, 8, 64, 32, "all_in"), id="all_in_vocab"),
        pytest.param((2, 8, 64, 32, "all_out"), id="all_out_vocab"),
        pytest.param((2, 9, 8, 4, "default"), id="ragged_tiny_vocab"),
    ],
)
def test_streamed_backward_edge_cases(tp_group, case):
    """Edge cases for the streamed buffer: short sequences, mask extremes, ragged tiling.

    ``seq_len=1`` writes a single one-element slice; an all-False or all-True
    ``target_mask`` exercises the empty/full ``scatter_add_`` path; ``seq_len=9``
    with ``chunk_size=4`` yields three chunks whose final slice ``[8:9]`` is the
    clamped ragged remainder. Each must still match the single-shot grad exactly.
    """
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        ChunkedDistributedLogprob,
        DistributedLogprob,
    )

    batch_size, seq_len, vocab_size, chunk_size, mask_mode = case
    device = torch.device("cpu")
    torch.manual_seed(1)

    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32, device=device) * 2.0
    if mask_mode == "all_out":
        target = torch.full((batch_size, seq_len), vocab_size + 5, device=device, dtype=torch.long)
    else:
        target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    grad_ref = _backward_grad(DistributedLogprob, logits, target, 0, vocab_size, tp_group)
    grad_chunk = _backward_grad(
        ChunkedDistributedLogprob,
        logits,
        target,
        0,
        vocab_size,
        tp_group,
        chunk_size=chunk_size,
    )

    assert torch.equal(grad_chunk, grad_ref), "streamed chunked grad must be bit-identical to non-chunked grad"


def test_streamed_backward_covers_sequence_without_overlap(tp_group):
    """The streamed buffer is fully written exactly once: tiling covers [0, seq_size).

    Prime ``seq_len`` (17) with ``chunk_size=5`` and ``batch_size>1`` gives four
    chunks whose final slice ``[15:17]`` is the clamped ragged remainder, so the
    tiling does not evenly divide the sequence. ``grad_input`` is a ``torch.empty``
    buffer (uninitialised), so any off-by-one in the chunk tiling (a position
    never written) would leave that slice holding pre-existing memory.

    Rather than rely on ``torch.isfinite`` -- ``torch.empty`` can return finite
    fp32 garbage, so an unwritten slice need not surface as NaN/inf -- we compare
    the full streamed grad bit-for-bit against the single-shot ``DistributedLogprob``
    reference. An unwritten position would hold the reference value only by
    astronomically unlikely coincidence, so ``torch.equal`` over the whole tensor
    is a strict full-coverage check for this ragged/prime configuration (the main
    sweep asserts byte-identity but only at ``seq_len=30``).
    """
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        ChunkedDistributedLogprob,
        DistributedLogprob,
    )

    device = torch.device("cpu")
    torch.manual_seed(2)

    batch_size = 3
    seq_len = 17  # prime length -> ragged final chunk for every chunk_size > 1
    vocab_size = 128
    chunk_size = 5

    logits = torch.randn(batch_size, seq_len, vocab_size, dtype=torch.float32, device=device) * 2.0
    target = torch.randint(0, vocab_size, (batch_size, seq_len), device=device, dtype=torch.long)

    grad_ref = _backward_grad(DistributedLogprob, logits, target, 0, vocab_size, tp_group)
    grad_chunk = _backward_grad(
        ChunkedDistributedLogprob,
        logits,
        target,
        0,
        vocab_size,
        tp_group,
        chunk_size=chunk_size,
    )

    assert grad_chunk.shape == grad_ref.shape == logits.shape
    # Bit-identity over every position is a strict full-coverage check: an
    # unwritten torch.empty slice cannot coincidentally equal the reference.
    assert torch.equal(grad_chunk, grad_ref), "every sequence slice of the preallocated buffer must be written"
