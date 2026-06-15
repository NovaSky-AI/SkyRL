"""Real TP>1 vocab-parallel parity for the streamed-buffer ChunkedDistributedLogprob.backward.

``ChunkedDistributedLogprob.backward`` streams each chunk's gradient straight
into one preallocated ``[batch_size, seq_size, partition_vocab_size]`` fp32
buffer instead of appending per-chunk grads to a Python list and ``torch.cat``-ing
them. The refactor is meant to be byte-identical. ``test_chunked_logprob_backward.py``
covers the single-rank (world_size=1) path; this test covers the path the change
actually matters on: the *distributed vocab-parallel* backward at ``TP>1``, where
each rank owns ``vocab_size // TP`` columns of logits and the chunk loop writes
into a per-rank ``partition_vocab_size``-wide buffer.

Strategy: spawn ``TP`` NCCL ranks, initialise Megatron model-parallel with
``tensor_model_parallel_size=TP``, shard a shared full-vocab logits tensor across
ranks, and run ``ChunkedDistributedLogprob`` forward+backward. Each rank's local
grad slice (its ``vocab_size // TP`` columns) must match the corresponding columns
of a single-process, full-vocab autograd reference computed with the same fp32
log-softmax math — within fp32 tolerance (``torch.testing.assert_close``), with no
chunk overlap (the union of slices tiles the full vocab exactly once).

Requires ``TP`` free GPUs. It will NOT run on a CPU-only / macOS dev box.

Run with (>=2 free GPUs; the ``tp_size=4`` case is skipped unless 4 are present):
  uv run --isolated --extra dev --extra megatron -- \
    pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_chunked_logprob_backward_tp.py
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Shared problem definition: every rank must build the SAME full-vocab tensors,
# so seeds and shapes are module-level constants rather than per-rank randoms.
_SEED = 0
_BATCH = 4
_SEQ_LEN = 30  # not a multiple of the chunk sizes below -> ragged final chunk
_VOCAB = 256
_CHUNK_SIZE = 7


def _reference_full_grad(logits_fp32: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Single-process, full-vocab autograd reference for the chosen-token logprob grad.

    Computes ``log_softmax`` over the full vocabulary and gathers the target
    logprob, exactly the math the distributed path reproduces across TP ranks
    (with the cross-rank all-reduce collapsing to the full-vocab reduction here).
    Returns the grad of ``sum(grad_seed * logprob)`` w.r.t. the logits.
    """
    leaf = logits_fp32.detach().clone().requires_grad_(True)
    log_probs = torch.log_softmax(leaf, dim=-1)
    chosen = torch.gather(log_probs, -1, target.unsqueeze(-1)).squeeze(-1)
    grad_seed = torch.linspace(0.5, 1.5, steps=chosen.numel(), device=chosen.device, dtype=chosen.dtype).reshape(
        chosen.shape
    )
    (grad_seed * chosen).sum().backward()
    return leaf.grad.detach()


def _set_ci_nccl_env():
    """Mirror the NCCL env the rest of the megatron CI relies on (conftest._build_ray_env_vars).

    The gpu_ci conftest sets these inside Ray's runtime_env, but ``mp.spawn``
    children inherit none of them. Without ``NCCL_CUMEM_ENABLE=0`` NCCL 2.28's
    cuMem-based commAlloc SEGVs on the CI driver, and the megatron path needs
    ``CUDA_DEVICE_MAX_CONNECTIONS=1``; P2P/SHM are disabled when peer access is
    unsupported (same guard conftest uses). Set before ``init_process_group``.
    """
    from skyrl.train.utils.utils import run_p2p_access_check

    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    # Each spawned rank is already pinned to its own GPU, so check peer access
    # directly rather than via peer_access_supported (which would spin up Ray).
    if not run_p2p_access_check():
        os.environ["NCCL_P2P_DISABLE"] = "1"
        os.environ["NCCL_SHM_DISABLE"] = "1"


def _tp_worker(rank: int, world_size: int, master_port: str, result_path: str):
    """One TP rank: shard the vocab, run chunked forward+backward, save the local grad."""
    import megatron.core.parallel_state as mpu

    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        ChunkedDistributedLogprob,
    )

    torch.cuda.set_device(rank)
    # Set the CI NCCL env before init_process_group; mp.spawn children inherit
    # none of the gpu_ci conftest runtime_env vars (see _set_ci_nccl_env).
    _set_ci_nccl_env()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # tensor_model_parallel_size == world_size -> a single TP group spanning all ranks.
    mpu.initialize_model_parallel(tensor_model_parallel_size=world_size)
    tp_group = mpu.get_tensor_model_parallel_group()

    device = torch.device("cuda", rank)
    torch.manual_seed(_SEED)

    # Every rank reconstructs the identical full-vocab logits/targets, then keeps
    # only its own vocab shard. partition_vocab_size = _VOCAB // world_size.
    assert _VOCAB % world_size == 0
    partition = _VOCAB // world_size
    vocab_start = rank * partition
    vocab_end = vocab_start + partition

    logits_full = (torch.randn(_BATCH, _SEQ_LEN, _VOCAB, dtype=torch.float32, device=device) * 2.0).contiguous()
    target = torch.randint(0, _VOCAB, (_BATCH, _SEQ_LEN), device=device, dtype=torch.long)

    leaf = logits_full[:, :, vocab_start:vocab_end].detach().clone().requires_grad_(True)
    out = ChunkedDistributedLogprob.apply(leaf, target, vocab_start, vocab_end, _CHUNK_SIZE, tp_group, False)
    grad_seed = torch.linspace(0.5, 1.5, steps=out.numel(), device=out.device, dtype=out.dtype).reshape(out.shape)
    out.backward(grad_seed)

    # Save the local grad shard plus the inputs so rank 0 can build the reference.
    torch.save(
        {
            "rank": rank,
            "vocab_start": vocab_start,
            "vocab_end": vocab_end,
            "grad_local": leaf.grad.detach().cpu(),
            "logits_full": logits_full.detach().cpu(),
            "target": target.detach().cpu(),
            "logprob_out": out.detach().cpu(),
        },
        f"{result_path}.{rank}",
    )

    mpu.destroy_model_parallel()
    dist.destroy_process_group()


@pytest.mark.megatron
@pytest.mark.parametrize("tp_size", [2, 4])
def test_streamed_chunked_backward_matches_reference_at_tp(tmp_path, tp_size):
    """Per-rank streamed chunked grad equals the full-vocab reference, with no overlap.

    Spawns ``tp_size`` NCCL ranks; each owns ``_VOCAB // tp_size`` vocab columns.
    The full-vocab single-process reference grad is sliced per rank and compared
    within fp32 tolerance (``torch.testing.assert_close``) against that rank's
    streamed chunked grad. The union of the rank vocab slices is asserted to tile
    ``[0, _VOCAB)`` exactly once (no gap, no overlap), which is the distributed
    analogue of the chunk loop tiling ``[0, seq_size)`` exactly.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the vocab-parallel backward")
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"requires {tp_size} GPUs, found {torch.cuda.device_count()}")

    from skyrl.backends.skyrl_train.distributed.utils import get_free_port

    master_port = str(get_free_port())
    result_path = str(tmp_path / "tp_grad")

    mp.spawn(_tp_worker, args=(tp_size, master_port, result_path), nprocs=tp_size, join=True)

    shards = [torch.load(f"{result_path}.{rank}") for rank in range(tp_size)]

    # All ranks must have built the identical full-vocab problem.
    logits_full = shards[0]["logits_full"]
    target = shards[0]["target"]
    for shard in shards[1:]:
        assert torch.equal(shard["logits_full"], logits_full)
        assert torch.equal(shard["target"], target)

    grad_ref = _reference_full_grad(logits_full, target)

    # No-overlap / full-coverage: the rank vocab slices tile [0, _VOCAB) exactly once.
    covered = torch.zeros(_VOCAB, dtype=torch.int64)
    for shard in shards:
        covered[shard["vocab_start"] : shard["vocab_end"]] += 1
    assert torch.equal(covered, torch.ones(_VOCAB, dtype=torch.int64)), "vocab slices must tile [0, vocab) once"

    # Per-rank grad matches the corresponding reference columns within fp32 tolerance
    # (the cross-rank all-reduce reorders the fp32 reduction vs. the single-tensor reference).
    for shard in shards:
        ref_slice = grad_ref[:, :, shard["vocab_start"] : shard["vocab_end"]]
        assert shard["grad_local"].shape == ref_slice.shape
        torch.testing.assert_close(shard["grad_local"], ref_slice, atol=1e-5, rtol=1e-4)
