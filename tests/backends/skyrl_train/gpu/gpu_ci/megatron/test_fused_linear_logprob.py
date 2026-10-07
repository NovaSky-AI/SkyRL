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
    _compute_distributed_log_softmax,
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


def _fused_entropy_fb(hidden, weight, target, vstart, vend, tp_group, chunk_size):
    """Return log-prob, no-grad entropy, and log-prob-only gradients."""
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    out, entropy = FusedLinearChunkedDistributedLogprob.apply(
        h, w, target, vstart, vend, chunk_size, tp_group, False, True
    )
    out.backward(_grad_seed(out))
    return out.detach(), entropy, h.grad.detach(), w.grad.detach()


def _fused_entropy_loss_fb(hidden, weight, target, vstart, vend, tp_group, chunk_size):
    """Return fused outputs and gradients from both log-prob and entropy."""
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    out, entropy = FusedLinearChunkedDistributedLogprob.apply(
        h, w, target, vstart, vend, chunk_size, tp_group, False, True, True
    )
    entropy_seed = torch.linspace(-0.25, 0.75, steps=entropy.numel(), device=entropy.device).reshape(entropy.shape)
    ((out * _grad_seed(out)).sum() + (entropy * entropy_seed).sum()).backward()
    return out.detach(), entropy.detach(), h.grad.detach(), w.grad.detach()


def _materialized_entropy_loss_fb(hidden, weight, target):
    """Full-vocabulary autograd reference for the entropy-loss path."""
    h = hidden.detach().clone().requires_grad_(True)
    w = weight.detach().clone().requires_grad_(True)
    logits = torch.matmul(h.to(w.dtype), w.t()).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    out = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    entropy_seed = torch.linspace(-0.25, 0.75, steps=entropy.numel(), device=entropy.device).reshape(entropy.shape)
    ((out * _grad_seed(out)).sum() + (entropy * entropy_seed).sum()).backward()
    return out.detach(), entropy.detach(), h.grad.detach(), w.grad.detach()


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


def test_fused_reuses_projection_for_no_grad_entropy(tp_group):
    device = torch.device("cuda")
    torch.manual_seed(122)
    B, S, V = 2, 31, 4096
    hidden = torch.randn(B, S, H, dtype=torch.bfloat16, device=device)
    weight = torch.randn(V, H, dtype=torch.bfloat16, device=device) * (H**-0.5)
    target = torch.randint(0, V, (B, S), device=device, dtype=torch.long)

    out, entropy, grad_hidden, grad_weight = _fused_entropy_fb(hidden, weight, target, 0, V, tp_group, 7)
    baseline_out, baseline_grad_hidden, baseline_grad_weight = _fused_fb(hidden, weight, target, 0, V, tp_group, 7)
    logits = hidden @ weight.T
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    expected_entropy = -(log_probs.exp() * log_probs).sum(dim=-1)

    assert not entropy.requires_grad
    torch.testing.assert_close(out, baseline_out, atol=0, rtol=0)
    torch.testing.assert_close(entropy, expected_entropy, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(grad_hidden, baseline_grad_hidden, atol=0, rtol=0)
    torch.testing.assert_close(grad_weight, baseline_grad_weight, atol=0, rtol=0)


def test_fused_entropy_loss_matches_materialized_logits(tp_group):
    device = torch.device("cuda")
    torch.manual_seed(123)
    B, S, V = 2, 31, 4096
    hidden = torch.randn(B, S, H, dtype=torch.bfloat16, device=device)
    weight = torch.randn(V, H, dtype=torch.bfloat16, device=device) * (H**-0.5)
    target = torch.randint(0, V, (B, S), device=device, dtype=torch.long)

    actual = _fused_entropy_loss_fb(hidden, weight, target, 0, V, tp_group, 7)
    expected = _materialized_entropy_loss_fb(hidden, weight, target)
    torch.testing.assert_close(actual[0], expected[0], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual[1], expected[1], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(actual[2], expected[2], atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(actual[3].float(), expected[3].float(), atol=4e-2, rtol=4e-2)


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

        out_e, entropy, gh_e, gw_e = _fused_entropy_fb(hidden, weight, target, vstart, vend, tp_group, chunk_size)
        logits = hidden.float() @ weight.float().T
        distributed_log_probs = _compute_distributed_log_softmax(logits, group=tp_group)
        expected_entropy = -(distributed_log_probs.exp() * distributed_log_probs).sum(dim=-1)
        dist.all_reduce(expected_entropy)
        torch.testing.assert_close(out_e, out_f, atol=0, rtol=0)
        torch.testing.assert_close(entropy, expected_entropy, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(gh_e, gh_f, atol=0, rtol=0)
        torch.testing.assert_close(gw_e, gw_f, atol=0, rtol=0)

        out_el, entropy_el, gh_el, gw_el = _fused_entropy_loss_fb(
            hidden, weight, target, vstart, vend, tp_group, chunk_size
        )
        full_weight = w_full.to(dev, weight.dtype)
        out_er, entropy_er, gh_er, gw_er = _materialized_entropy_loss_fb(hidden, full_weight, target)
        torch.testing.assert_close(out_el, out_er, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(entropy_el, entropy_er, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(gh_el, gh_er, atol=5e-3, rtol=5e-3)
        torch.testing.assert_close(gw_el.float(), gw_er[vstart:vend].float(), atol=5e-3, rtol=5e-3)

    ok = torch.tensor([1.0], device=dev)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    if rank == 0:
        print(f"RESULT: PASS (TP={world})")
    dist.destroy_process_group()


def _distributed_packed_cp_entropy_main():
    """Check differentiable fused entropy through the packed TP2/CP2 layout."""
    from skyrl.backends.skyrl_train.distributed.megatron.active_spans import (
        build_packed_active_metadata,
    )
    from skyrl.backends.skyrl_train.distributed.megatron.model_utils import (
        _get_tokens_on_this_cp_rank,
        from_parallel_hidden_to_logprobs_packed_sequences,
    )
    from skyrl.train.fused_lm_head import FusedLmHeadBackend

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    tp_size = cp_size = 2
    tp_rank = rank % tp_size
    cp_rank = rank // tp_size
    tp_groups = [dist.new_group([cp * tp_size + tp for tp in range(tp_size)]) for cp in range(cp_size)]
    cp_groups = [dist.new_group([cp * tp_size + tp for cp in range(cp_size)]) for tp in range(tp_size)]
    tp_group = tp_groups[cp_rank]
    cp_group = cp_groups[tp_rank]

    batch, sequence, hidden_size, vocab = 1, 64, H, 4096
    generator = torch.Generator().manual_seed(4321)
    full_hidden = torch.randn(batch, sequence, hidden_size, generator=generator, dtype=torch.float32).to(
        device, torch.bfloat16
    )
    full_weight = (torch.randn(vocab, hidden_size, generator=generator) * hidden_size**-0.5).to(device)
    target = torch.randint(0, vocab, (batch, sequence), generator=generator).to(device)
    loss_mask = torch.zeros((batch, sequence - 1), dtype=torch.bool)
    loss_mask[:, 8:29] = True
    loss_mask[:, 37:] = True
    metadata = build_packed_active_metadata(
        loss_mask,
        num_actions=sequence - 1,
        sequence_length=sequence,
        attention_mask=torch.ones((batch, sequence), dtype=torch.bool),
        sub_seq_lengths=[[sequence]],
        tp_size=tp_size,
        cp_size=cp_size,
        cp_rank=cp_rank,
        fp8_enabled=False,
    )
    local_hidden_source = _get_tokens_on_this_cp_rank(full_hidden, cp_rank, cp_size, seq_dim=1)
    local_weight_source = full_weight[tp_rank * (vocab // tp_size) : (tp_rank + 1) * (vocab // tp_size)]
    cu_seqlens = torch.tensor([0, sequence], device=device, dtype=torch.int32)
    device_loss_mask = loss_mask.to(device)
    logprob_seed = torch.linspace(0.5, 1.5, sequence - 1, device=device).unsqueeze(0)
    entropy_seed = torch.linspace(-0.25, 0.75, sequence - 1, device=device).unsqueeze(0)

    for backend in (
        FusedLmHeadBackend.TORCH,
        FusedLmHeadBackend.TRITON,
        FusedLmHeadBackend.TRITON_BLOCK_SPARSE,
    ):
        hidden = local_hidden_source.detach().clone().requires_grad_(True)
        weight = local_weight_source.detach().clone().requires_grad_(True)
        block_sparse = backend == FusedLmHeadBackend.TRITON_BLOCK_SPARSE
        logprobs, entropy = from_parallel_hidden_to_logprobs_packed_sequences(
            hidden,
            weight,
            target,
            cu_seqlens,
            sequence,
            vocab_start_index=tp_rank * (vocab // tp_size),
            vocab_end_index=(tp_rank + 1) * (vocab // tp_size),
            group=tp_group,
            cp_group=cp_group,
            chunk_size=16,
            attention_mask=torch.ones((batch, sequence), dtype=torch.bool, device=device),
            sub_seq_lengths=[[sequence]],
            fused_backend=backend,
            active_mask=metadata.active_mask.to(device) if block_sparse else None,
            active_spans=metadata.active_spans if block_sparse else None,
            return_entropy=True,
            entropy_requires_grad=True,
        )
        local_loss = ((logprobs * logprob_seed + entropy * entropy_seed) * device_loss_mask).sum()
        local_loss.backward()
        dist.all_reduce(hidden.grad, group=tp_group)
        dist.all_reduce(weight.grad, group=cp_group)

        reference_hidden = full_hidden.detach().clone().requires_grad_(True)
        reference_weight = full_weight.detach().clone().requires_grad_(True)
        reference_logits = reference_hidden.float() @ reference_weight.T
        reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
        reference_selected = reference_log_probs[:, :-1].gather(-1, target[:, 1:].unsqueeze(-1)).squeeze(-1)
        reference_entropy = -(reference_log_probs[:, :-1].exp() * reference_log_probs[:, :-1]).sum(dim=-1)
        ((reference_selected * logprob_seed + reference_entropy * entropy_seed) * device_loss_mask).sum().backward()
        expected_hidden_grad = _get_tokens_on_this_cp_rank(reference_hidden.grad, cp_rank, cp_size, seq_dim=1)
        expected_weight_grad = reference_weight.grad[tp_rank * (vocab // tp_size) : (tp_rank + 1) * (vocab // tp_size)]

        expected_logprobs = (
            reference_selected if not block_sparse else reference_selected.masked_fill(~device_loss_mask, 0.0)
        )
        expected_entropy = (
            reference_entropy if not block_sparse else reference_entropy.masked_fill(~device_loss_mask, 0.0)
        )
        torch.testing.assert_close(logprobs, expected_logprobs, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(entropy, expected_entropy, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(hidden.grad, expected_hidden_grad, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(weight.grad, expected_weight_grad, atol=2e-2, rtol=2e-2)

    if rank == 0:
        print("RESULT: PASS (packed TP=2 CP=2 entropy gradients)")
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


def test_fused_entropy_loss_packed_tp2_cp2():
    if torch.cuda.device_count() < 4:
        pytest.skip(f"needs >= 4 GPUs, have {torch.cuda.device_count()}")
    env = dict(
        os.environ,
        MASTER_ADDR="localhost",
        MASTER_PORT=str(get_free_port()),
        SKYRL_TEST_PACKED_CP_ENTROPY="1",
    )
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=4",
        "--master_port",
        env["MASTER_PORT"],
        __file__,
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
    assert "RESULT: PASS (packed TP=2 CP=2 entropy gradients)" in (
        result.stdout + result.stderr
    ), f"packed TP2/CP2 failed:\n{result.stdout}\n{result.stderr}"


if __name__ == "__main__":
    if os.environ.get("SKYRL_TEST_PACKED_CP_ENTROPY") == "1":
        _distributed_packed_cp_entropy_main()
    else:
        _distributed_main()
