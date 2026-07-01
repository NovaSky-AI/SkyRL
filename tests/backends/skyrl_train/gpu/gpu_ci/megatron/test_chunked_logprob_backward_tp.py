"""TP>1 parity for streamed ``ChunkedDistributedLogprob.backward``.

Spawns NCCL ranks, shards a shared full-vocab problem, and compares each local
grad shard with a single-process fp32 reference.

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

# Every rank rebuilds the same full-vocab problem.
_SEED = 0
_BATCH = 4
_SEQ_LEN = 30
_VOCAB = 256
_CHUNK_SIZE = 7


def _reference_full_grad(logits_fp32: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Full-vocab reference grad for chosen-token logprobs."""
    leaf = logits_fp32.detach().clone().requires_grad_(True)
    log_probs = torch.log_softmax(leaf, dim=-1)
    chosen = torch.gather(log_probs, -1, target.unsqueeze(-1)).squeeze(-1)
    grad_seed = torch.linspace(0.5, 1.5, steps=chosen.numel(), device=chosen.device, dtype=chosen.dtype).reshape(
        chosen.shape
    )
    (grad_seed * chosen).sum().backward()
    return leaf.grad.detach()


def _set_ci_nccl_env():
    """Apply the gpu_ci NCCL env that ``mp.spawn`` children do not inherit."""
    from skyrl.train.utils.utils import run_p2p_access_check

    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    os.environ["NVTE_FUSED_ATTN"] = "0"
    # Avoid peer_access_supported here; it would spin up Ray.
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
    # Set before init_process_group; spawned children miss the conftest runtime_env.
    _set_ci_nccl_env()

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    mpu.initialize_model_parallel(tensor_model_parallel_size=world_size)
    tp_group = mpu.get_tensor_model_parallel_group()

    device = torch.device("cuda", rank)
    torch.manual_seed(_SEED)

    # Build the same full-vocab problem on every rank, then keep this rank's shard.
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

    # Rank 0 later builds the reference from these shared inputs.
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
    """Per-rank streamed grads match the full-vocab reference slices."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for the vocab-parallel backward")
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"requires {tp_size} GPUs, found {torch.cuda.device_count()}")

    from skyrl.backends.skyrl_train.distributed.utils import get_free_port

    master_port = str(get_free_port())
    result_path = str(tmp_path / "tp_grad")

    mp.spawn(_tp_worker, args=(tp_size, master_port, result_path), nprocs=tp_size, join=True)

    shards = [torch.load(f"{result_path}.{rank}") for rank in range(tp_size)]

    # Sanity-check that the spawned ranks used the same problem.
    logits_full = shards[0]["logits_full"]
    target = shards[0]["target"]
    for shard in shards[1:]:
        assert torch.equal(shard["logits_full"], logits_full)
        assert torch.equal(shard["target"], target)

    grad_ref = _reference_full_grad(logits_full, target)

    # Rank vocab slices must tile [0, _VOCAB) exactly once.
    covered = torch.zeros(_VOCAB, dtype=torch.int64)
    for shard in shards:
        covered[shard["vocab_start"] : shard["vocab_end"]] += 1
    assert torch.equal(covered, torch.ones(_VOCAB, dtype=torch.int64)), "vocab slices must tile [0, vocab) once"

    # Allow fp32 reduction-order noise from the cross-rank all-reduce.
    for shard in shards:
        ref_slice = grad_ref[:, :, shard["vocab_start"] : shard["vocab_end"]]
        assert shard["grad_local"].shape == ref_slice.shape
        torch.testing.assert_close(shard["grad_local"], ref_slice, atol=1e-5, rtol=1e-4)
