"""Equivalency + parallelism tests for the fused-linear (Liger-style) log-prob.

``FusedLinearChunkedDistributedLogprob`` folds the LM-head matmul into the
chunked, TP/CP-parallel token-logprob so the full ``[B, S, vocab//TP]`` logits
(and their fp32 gradient) are never materialized. These tests assert it is
numerically identical — forward log-probs *and* both gradients (``grad_hidden``,
``grad_weight``) — to the existing materialize-logits path
(``hidden @ weightᵀ`` -> ``ChunkedDistributedLogprob``), across:

  * chunk sizes (incl. > seq_len), out-of-vocab targets, edge shapes,
  * mixed dtypes (bf16 hidden + fp32 weight — the real Megatron case),
  * tensor/vocab parallelism: TP=1 (here) and TP>1 (spawned via torchrun).

It also checks the forward against Liger's own fused-linear-CE as an oracle
(``logprob == -LigerFLCE(reduction="none")``) when liger-kernel is installed.

TP=1 (single process):
  uv run --isolated --extra dev --extra megatron -- \
    pytest -s tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_fused_linear_logprob.py
TP>1 is launched automatically as a torchrun subprocess by ``test_fused_linear_logprob_tp``.
"""

import os
import subprocess
import sys

import pytest
import torch
import torch.distributed as dist

from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
    ChunkedDistributedLogprob,
    FusedLinearChunkedDistributedLogprob,
)
from skyrl.train.utils.utils import get_free_port

# Run as part of the Megatron GPU CI suite (`-m megatron`, --extra megatron).
pytestmark = pytest.mark.megatron

H = 256  # hidden size for the unit tests


@pytest.fixture(scope="module")
def tp_group():
    """Single-rank TP process group (world_size=1; all-reduces are no-ops)."""
    if not dist.is_initialized():
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(get_free_port())
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield dist.group.WORLD
    if dist.is_initialized():
        dist.destroy_process_group()


def _grad_seed(out):
    # Non-uniform upstream gradient so any per-position bug surfaces.
    return torch.linspace(0.5, 1.5, steps=out.numel(), device=out.device, dtype=out.dtype).reshape(out.shape)


def _fused_fb(hidden, weight, target, vstart, vend, tp_group, chunk_size):
    """Forward+backward through the fused op -> (logprob, grad_hidden, grad_weight)."""
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    out = FusedLinearChunkedDistributedLogprob.apply(h, w, target, vstart, vend, chunk_size, tp_group, False)
    out.backward(_grad_seed(out))
    return out.detach(), h.grad.detach(), w.grad.detach()


def _reference_fb(hidden, weight, target, vstart, vend, tp_group, chunk_size):
    """Materialize logits = hidden @ weightᵀ then the validated ChunkedDistributedLogprob.

    Casting hidden to the weight dtype mirrors the ColumnParallelLinear output
    layer; autograd then yields grad_hidden / grad_weight to compare against.
    """
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    logits = torch.matmul(h.to(w.dtype), w.t())
    out = ChunkedDistributedLogprob.apply(logits, target, vstart, vend, chunk_size, tp_group, False)
    out.backward(_grad_seed(out))
    return out.detach(), h.grad.detach(), w.grad.detach()


def _assert_equivalent(hidden, weight, target, vstart, vend, tp_group, chunk_size):
    out_f, gh_f, gw_f = _fused_fb(hidden, weight, target, vstart, vend, tp_group, chunk_size)
    out_r, gh_r, gw_r = _reference_fb(hidden, weight, target, vstart, vend, tp_group, chunk_size)
    assert out_f.dtype == torch.float32
    torch.testing.assert_close(out_f, out_r, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(gh_f, gh_r, atol=2e-3, rtol=2e-3)
    # grad_weight accumulates over all B*S tokens; the fused op sums in fp32 in a
    # different order than autograd's matmul backward, so a bf16 weight needs a
    # looser (still firmly bug-catching) bound while fp32 stays tight.
    gw_tol = 4e-2 if weight.dtype == torch.bfloat16 else 2e-3
    torch.testing.assert_close(gw_f.float(), gw_r.float(), atol=gw_tol, rtol=gw_tol)


@pytest.mark.parametrize("chunk_size", [1, 7, 64, 512])
@pytest.mark.parametrize("with_oov_targets", [False, True])
@pytest.mark.parametrize(
    "hidden_dtype, weight_dtype",
    [(torch.bfloat16, torch.bfloat16), (torch.bfloat16, torch.float32), (torch.float32, torch.float32)],
)
def test_fused_matches_materialized_logits(tp_group, chunk_size, with_oov_targets, hidden_dtype, weight_dtype):
    """Fused op == materialize-logits reference (fwd + grad_hidden + grad_weight).

    The (bf16 hidden, fp32 weight) combo is the real Megatron case and guards the
    dtype-promotion path in the fused matmul.
    """
    device = torch.device("cuda")
    torch.manual_seed(0)
    B, S, V = 4, 32, 4096
    target_high = V + 1024 if with_oov_targets else V

    hidden = torch.randn(B, S, H, dtype=hidden_dtype, device=device)
    weight = torch.randn(V, H, dtype=weight_dtype, device=device) * (H**-0.5)
    target = torch.randint(0, target_high, (B, S), device=device, dtype=torch.long)

    _assert_equivalent(hidden, weight, target, 0, V, tp_group, chunk_size)


@pytest.mark.parametrize(
    "case",
    [
        pytest.param((1, 1, 1024, 4, "default"), id="seq1"),
        pytest.param((2, 8, 1024, 32, "all_in"), id="all_in_vocab"),
        pytest.param((2, 8, 1024, 32, "all_out"), id="all_out_vocab"),
        pytest.param((2, 8, 8, 4, "default"), id="tiny_vocab"),
    ],
)
def test_fused_matches_materialized_logits_edge_cases(tp_group, case):
    B, S, V, chunk_size, mask_mode = case
    device = torch.device("cuda")
    torch.manual_seed(1)
    hidden = torch.randn(B, S, H, dtype=torch.bfloat16, device=device)
    weight = torch.randn(V, H, dtype=torch.float32, device=device) * (H**-0.5)
    if mask_mode == "all_out":
        target = torch.full((B, S), V + 5, device=device, dtype=torch.long)
    else:
        target = torch.randint(0, V, (B, S), device=device, dtype=torch.long)
    _assert_equivalent(hidden, weight, target, 0, V, tp_group, chunk_size)


def test_fused_forward_matches_liger_flce(tp_group):
    """Oracle: at TP=1 the fused log-prob equals -LigerFLCE(reduction='none').

    Validates our kernel against Liger's own fused-linear-cross-entropy math.
    Forward only — Liger's FLCE does not support a reduction='none' backward.
    """
    liger = pytest.importorskip("liger_kernel.ops.fused_linear_cross_entropy")
    device = torch.device("cuda")
    torch.manual_seed(2)
    B, S, V = 2, 16, 4096
    hidden = torch.randn(B, S, H, dtype=torch.bfloat16, device=device)
    weight = torch.randn(V, H, dtype=torch.bfloat16, device=device) * (H**-0.5)
    target = torch.randint(0, V, (B, S), device=device, dtype=torch.long)

    out_f, _, _ = _fused_fb(hidden, weight, target, 0, V, tp_group, 64)

    flce = liger.LigerFusedLinearCrossEntropyFunction.apply
    ce = flce(hidden.reshape(B * S, H), weight, target.reshape(B * S), None, None, -100, 0.0, 0.0, "none")
    ce = ce[0] if isinstance(ce, tuple) else ce
    torch.testing.assert_close(out_f.reshape(-1), (-ce).float(), atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# TP>1: vocab-parallel correctness via torchrun (spawned as a subprocess).
# ---------------------------------------------------------------------------


def _distributed_main():
    """Run the fused-vs-reference equivalency under real tensor/vocab parallelism.

    Each rank owns a vocab shard [r*V/TP, (r+1)*V/TP); the cross-rank max /
    sum-exp / chosen-logit all-reduces are exercised. Identical inputs on every
    rank (seeded) so only the vocab dim is sharded.
    """
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device(f"cuda:{rank}")
    tp_group = dist.group.WORLD

    B, S, V = 2, 64, 4096
    assert V % world == 0
    v_tp = V // world
    vstart, vend = rank * v_tp, (rank + 1) * v_tp

    g = torch.Generator().manual_seed(1234)
    hidden = torch.randn(B, S, H, generator=g, dtype=torch.float32).to(dev, torch.bfloat16)
    target = torch.randint(0, V, (B, S), generator=g)
    w_full = torch.randn(V, H, generator=g, dtype=torch.float32) * (H**-0.5)
    weight = w_full[vstart:vend].to(dev, torch.float32)
    target = target.to(dev)

    for chunk_size in (16, 512):
        out_f, gh_f, gw_f = _fused_fb(hidden, weight, target, vstart, vend, tp_group, chunk_size)
        out_r, gh_r, gw_r = _reference_fb(hidden, weight, target, vstart, vend, tp_group, chunk_size)
        torch.testing.assert_close(out_f, out_r, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(gh_f, gh_r, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(gw_f.float(), gw_r.float(), atol=5e-3, rtol=5e-3)

    ok = torch.tensor([1.0], device=dev)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"RESULT: PASS (TP={world})")
    dist.destroy_process_group()


@pytest.mark.parametrize("nproc", [2, 4])
def test_fused_linear_logprob_tp(nproc):
    """Spawn torchrun --nproc_per_node=nproc to check vocab-parallel correctness."""
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
