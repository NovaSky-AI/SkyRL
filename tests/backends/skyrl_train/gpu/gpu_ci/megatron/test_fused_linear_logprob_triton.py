"""GPU parity tests for ``FusedLinearLogprobTriton``.

The tests drive the autograd Function directly and compare it with the
materialized-logits reference for log-probs and both grads across TP1/2, OOV
targets, fp32, and bf16.

fp32 cases force IEEE precision for tight tolerances; bf16 uses the production
TF32 path with looser tolerances.

    uv run --isolated --extra dev --extra megatron -- pytest -s \
        tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_fused_linear_logprob_triton.py
"""

import ast
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import triton

from skyrl.backends.skyrl_train.distributed.megatron import fused_linear_logprob_triton
from skyrl.backends.skyrl_train.distributed.megatron.fused_linear_logprob_triton import (
    TRITON_AVAILABLE,
    FusedLinearLogprobTriton,
)

# Selected by the megatron GPU CI job; skip when the Triton kernel cannot run.
pytestmark = [
    pytest.mark.megatron,
    pytest.mark.skipif(
        not (torch.cuda.is_available() and TRITON_AVAILABLE),
        reason="Triton fused LM-head log-prob requires a CUDA device and triton",
    ),
]


def _kernel_source() -> str:
    return Path(fused_linear_logprob_triton.__file__).read_text()


def _load_schedule_symbols() -> dict[str, object]:
    module = ast.parse(_kernel_source())
    wanted = {
        "_AUTOTUNE_MIN_TOKEN_BUCKET",
        "_AUTOTUNE_MAX_TOKEN_BUCKET",
        "_FORWARD_MAINLOOP_CONFIG_SPECS",
        "_EPILOGUE_CONFIG_SPECS",
        "_EPILOGUE_UPDATE_CONFIG_SPECS",
        "_BACKWARD_CONFIG_SPECS",
        "_autotune_token_bucket",
    }
    nodes = [
        node
        for node in module.body
        if (isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in wanted)
        or (isinstance(node, ast.FunctionDef) and node.name in wanted)
    ]
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(nodes, type_ignores=[]), filename="<schedule>", mode="exec"),
        namespace,
    )
    return namespace


_SCHEDULE_SYMBOLS = _load_schedule_symbols()
_FORWARD_MAINLOOP_CONFIG_SPECS = _SCHEDULE_SYMBOLS["_FORWARD_MAINLOOP_CONFIG_SPECS"]
_EPILOGUE_CONFIG_SPECS = _SCHEDULE_SYMBOLS["_EPILOGUE_CONFIG_SPECS"]
_EPILOGUE_UPDATE_CONFIG_SPECS = _SCHEDULE_SYMBOLS["_EPILOGUE_UPDATE_CONFIG_SPECS"]
_BACKWARD_CONFIG_SPECS = _SCHEDULE_SYMBOLS["_BACKWARD_CONFIG_SPECS"]
_autotune_token_bucket = _SCHEDULE_SYMBOLS["_autotune_token_bucket"]


def test_autotune_schedules_include_incumbents_and_variants() -> None:
    assert (128, 256, 32, 5, 8) in _FORWARD_MAINLOOP_CONFIG_SPECS
    assert (128, 256, 32, 3, 8) in _FORWARD_MAINLOOP_CONFIG_SPECS
    assert (16, 64, 4) in _EPILOGUE_CONFIG_SPECS
    assert (16, 4) in _EPILOGUE_UPDATE_CONFIG_SPECS
    assert (128, 256, 32, 16, 3, 8) in _BACKWARD_CONFIG_SPECS
    for specs in (
        _FORWARD_MAINLOOP_CONFIG_SPECS,
        _EPILOGUE_CONFIG_SPECS,
        _EPILOGUE_UPDATE_CONFIG_SPECS,
        _BACKWARD_CONFIG_SPECS,
    ):
        assert len(specs) == len(set(specs))


def test_all_triton_entrypoints_use_bucketed_cached_autotune() -> None:
    module = ast.parse(_kernel_source())
    autotune_decorators = [
        decorator
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        for decorator in node.decorator_list
        if (
            isinstance(decorator, ast.Call)
            and isinstance(decorator.func, ast.Attribute)
            and decorator.func.attr == "autotune"
        )
    ]
    assert len(autotune_decorators) == 5
    for decorator in autotune_decorators:
        keywords = {keyword.arg: keyword.value for keyword in decorator.keywords}
        assert ast.literal_eval(keywords["cache_results"]) is True
        keys = ast.literal_eval(keywords["key"])
        assert "num_tokens_bucket" in keys
        assert "COMPUTE_ENTROPY" in keys


def test_skyrl_adapter_disables_unused_entropy() -> None:
    module = ast.parse(_kernel_source())
    adapter = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "FusedLinearLogprobTriton"
    )
    methods = {node.name: node for node in adapter.body if isinstance(node, ast.FunctionDef)}
    expected_calls = {
        "forward": "efficient_entropy_forward",
        "backward": "efficient_entropy_backward",
    }
    for method_name, call_name in expected_calls.items():
        calls = [
            node
            for node in ast.walk(methods[method_name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == call_name
        ]
        assert len(calls) == 1
        keywords = {keyword.arg: keyword.value for keyword in calls[0].keywords}
        assert isinstance(keywords["compute_entropy"], ast.Constant)
        assert keywords["compute_entropy"].value is False


def test_autotuned_epilogues_do_not_overwrite_inputs() -> None:
    module = ast.parse(_kernel_source())
    expected_output_arguments = {
        "efficient_entropy_triton_kernel_epilogue": {"result_logprobs_ptr"},
        "efficient_entropy_triton_epilogue_tp_update": {
            "result_entropy_b_ptr",
            "result_logprobs_ptr",
        },
    }
    for node in module.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in expected_output_arguments:
            continue
        argument_names = {argument.arg for argument in node.args.args}
        assert expected_output_arguments[node.name] <= argument_names


@pytest.mark.parametrize(
    ("num_tokens", "expected_bucket"),
    [
        (1, 128),
        (128, 128),
        (129, 256),
        (16384, 16384),
        (20000, 32768),
        (262144, 262144),
        (262145, 524288),
        (524288, 524288),
        (1048576, 1048576),
    ],
)
def test_autotune_token_bucket(num_tokens: int, expected_bucket: int) -> None:
    assert _autotune_token_bucket(num_tokens) == expected_bucket


def test_autotune_token_bucket_rejects_empty_input() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        _autotune_token_bucket(0)


def _direct_fused_logprobs(hidden, weight_shard, target_shifted, vstart, vend, chunk_size, grad_seed):
    """Run the Triton Function on already-shifted targets."""
    leaf_h = hidden.detach().clone().requires_grad_(True)
    leaf_w = weight_shard.detach().clone().requires_grad_(True)
    lp = FusedLinearLogprobTriton.apply(
        leaf_h,
        leaf_w,
        target_shifted,
        vstart,
        vend,
        chunk_size,
        dist.group.WORLD,
        False,
    )
    lp.backward(grad_seed.clone())
    return lp.detach(), leaf_h.grad.detach(), leaf_w.grad.detach()


def _stock_shifted(hidden, weight_shard, target_shifted, vstart, vend, chunk_size):
    """Materialized-logits reference on already-shifted targets."""
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
    # bf16 uses production TF32; fp32 forces IEEE precision.
    if dtype == torch.bfloat16:
        return dict(atol=2e-2, rtol=2e-2)
    return dict(atol=1e-4, rtol=1e-4)


def _worker(rank, world_size, port, chunk_size, with_oov, dtype_str, ret_dict):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # Only tiny label-logit / softmax-stat tensors cross ranks, so gloo is enough.
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    # mp.spawn starts a fresh interpreter, so set precision in each worker.
    _prev_force_ieee = fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION
    fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION = dtype_str == "fp32"
    try:
        torch.cuda.set_device(0)
        # L4/Ada cap per-block shared memory at ~99KB; the default 128x256 fp32 logits
        # tile needs 128KB. Fall back to a 128x128 tile only when the GPU can't fit the
        # production config, so A100/H100 CI still exercises the real tile.
        _smem = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
        if _smem < 128 * 256 * 4:
            fused_linear_logprob_triton.efficient_entropy_kernel_general_mainloop.configs = [
                triton.Config(
                    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
                    num_stages=2,
                    num_warps=4,
                )
            ]
        device = torch.device("cuda")
        dtype = torch.bfloat16 if dtype_str == "bf16" else torch.float32
        torch.manual_seed(0)  # identical across ranks => identical hidden/weight/target

        # verl requires hidden_size % 128 == 0.
        batch_size, seq_len, hidden_size, vocab_size = 3, 24, 128, 256
        hidden = (torch.randn(batch_size, seq_len, hidden_size, device=device) * 0.5).to(dtype)
        weight_full = (torch.randn(vocab_size, hidden_size, device=device) * 0.1).to(dtype)
        target_high = vocab_size + 50 if with_oov else vocab_size
        target = torch.randint(0, target_high, (batch_size, seq_len), device=device, dtype=torch.long)

        assert vocab_size % world_size == 0
        shard = vocab_size // world_size
        vstart, vend = rank * shard, (rank + 1) * shard
        weight_shard = weight_full[vstart:vend].contiguous()

        # Keep target shifting out of the kernel-under-test.
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

    TP2 covers vocab-parallel reductions and shard/OOV masking; fp32 uses tight
    IEEE tolerances, bf16 uses production precision.
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


@pytest.mark.parametrize("logit_offset", [-20.0, -120.0])
def test_epilogue_handles_strongly_negative_logits(logit_offset) -> None:
    """The epilogue's log-sum-exp shift must be the true row max, not max(0, max).

    A zero-initialised ``global_max`` leaves the shift at 0 whenever every logit for
    a token is negative, so ``exp(logit - 0)`` underflows to 0, ``log(accu)`` becomes
    -inf, and the log-prob comes back +inf with NaN grads. -20 is the control (it
    passes either way); -120 reproduces the failure.
    """
    device = torch.device("cuda")
    previous = fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION
    fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION = True
    try:
        num_tokens, hidden_size, vocab_size = 64, 128, 2048  # 2 forward splits
        gen = torch.Generator(device=device).manual_seed(0)
        hidden = torch.rand(num_tokens, hidden_size, device=device, generator=gen) + 0.5
        weight = torch.randn(vocab_size, hidden_size, device=device, generator=gen) * (hidden_size**-0.5)
        # Shift every logit of every token below zero (hidden is strictly positive).
        weight += logit_offset / hidden.sum(-1).min()
        labels = torch.randint(0, vocab_size, (num_tokens,), device=device, generator=gen)
        grad_logprobs = torch.linspace(0.5, 1.5, num_tokens, device=device)

        logits = hidden.double() @ weight.double().T
        assert logits.max().item() < 0.0, "setup must drive every logit negative"
        expected = logits.gather(1, labels[:, None]).squeeze(1) - torch.logsumexp(logits, dim=-1)

        logprobs, entropy, maximum, accumulate, entropy_b = fused_linear_logprob_triton.efficient_entropy_forward(
            hidden, weight, labels, 1.0, None
        )
        d_hidden, d_weight = fused_linear_logprob_triton.efficient_entropy_backward(
            grad_logprobs,
            torch.zeros_like(grad_logprobs),
            hidden,
            weight,
            labels,
            maximum,
            accumulate,
            entropy_b,
            False,
            1.0,
            None,
        )

        assert torch.isfinite(logprobs).all(), f"non-finite log-probs at max logit {logits.max().item():.1f}"
        assert torch.isfinite(entropy).all(), "non-finite entropy"
        assert torch.isfinite(d_hidden).all() and torch.isfinite(d_weight).all(), "non-finite grads"
        torch.testing.assert_close(logprobs.double(), expected, rtol=1e-3, atol=1e-3)
    finally:
        fused_linear_logprob_triton.FORCE_FP32_IEEE_PRECISION = previous
