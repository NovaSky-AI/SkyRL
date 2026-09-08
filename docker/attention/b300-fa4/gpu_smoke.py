"""Run isolated B300 attention cases and compare with an upstream FA4 environment."""

import argparse
import importlib
import importlib.metadata
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = list(itertools.product(("fa2", "fa4"), ("dense", "varlen"), (False, True), ((24, 4, 256), (32, 2, 128))))


def reference_attention(torch, q, k, v, causal, upcast):
    dtype = q.dtype
    q, k, v = [item.float() if upcast else item for item in (q, k, v)]
    groups = q.shape[-2] // k.shape[-2]
    k, v = [item.repeat_interleave(groups, dim=-2) for item in (k, v)]
    q, k, v = [item.transpose(-3, -2) for item in (q, k, v)]
    scale = q.shape[-1] ** -0.5
    scores = (q * scale) @ k.transpose(-2, -1) if upcast else q @ (k * scale).transpose(-2, -1)
    if causal:
        mask = torch.ones(scores.shape[-2:], dtype=torch.bool, device=q.device).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    return (scores.softmax(dim=-1) @ v).transpose(-3, -2).to(dtype)


def check_error(name, actual, reference, baseline):
    error = (actual - reference).abs().max().item()
    # Same error envelope as upstream tests/cute/test_flash_attn.py.
    tolerance = 2 * (baseline - reference).abs().max().item()
    tolerance += 2 * (reference + 0.3 - 0.3 - reference).abs().max().item()
    if error > tolerance:
        raise AssertionError(f"{name}: error {error} exceeds {tolerance}")
    return {"max_error": error, "tolerance": tolerance}


def run_case(index, output, oracle):
    torch = importlib.import_module("torch")
    torch.backends.cuda.matmul.allow_tf32 = False
    if torch.cuda.get_device_capability() != (10, 3):
        raise RuntimeError("Kernel qualification requires a B300 SM103 device")
    if oracle and importlib.metadata.version("flash-attn-4") != "4.0.0b28":
        raise RuntimeError("The reference environment must contain upstream FA4 4.0.0b28")
    backend, layout, causal, (heads, kv_heads, dimension) = CASES[index]
    interface = importlib.import_module("flash_attn" if backend == "fa2" else "flash_attn.cute.interface")
    torch.manual_seed(20260908)
    prefix = (2, 128) if layout == "dense" else (384,)
    originals = [
        torch.randn(*prefix, h, dimension, device="cuda", dtype=torch.bfloat16) for h in (heads, kv_heads, kv_heads)
    ]
    gradient = torch.randn(*prefix, heads, dimension, device="cuda", dtype=torch.bfloat16)
    statistics, result = [], None
    for iteration in range(2):
        q, k, v = [item.detach().clone().requires_grad_() for item in originals]
        kwargs = {"causal": causal, "deterministic": True}
        function = interface.flash_attn_func
        if layout == "varlen":
            cu = torch.tensor([0, 128, 384], device="cuda", dtype=torch.int32)
            kwargs.update(cu_seqlens_q=cu, cu_seqlens_k=cu, max_seqlen_q=256, max_seqlen_k=256)
            function = interface.flash_attn_varlen_func
        torch.cuda.synchronize()
        start = time.monotonic()
        out = function(q, k, v, **kwargs)
        if isinstance(out, tuple):
            out = out[0]
        gradients = torch.autograd.grad(out, (q, k, v), gradient)
        torch.cuda.synchronize()
        elapsed = time.monotonic() - start
        if not all(torch.isfinite(item).all().item() for item in (out, *gradients)):
            raise AssertionError("Attention produced nonfinite output or gradients")

        references = []
        for upcast in (True, False):
            inputs = [item.detach().clone().requires_grad_() for item in originals]
            if layout == "dense":
                ref = reference_attention(torch, *inputs, causal, upcast)
            else:
                ref = torch.cat(
                    [
                        reference_attention(
                            torch, *(item[left:right].unsqueeze(0) for item in inputs), causal, upcast
                        ).squeeze(0)
                        for left, right in ((0, 128), (128, 384))
                    ]
                )
            grads = torch.autograd.grad(ref, inputs, gradient)
            references.append((ref, *grads))
        errors = {
            name: check_error(name, actual, ref, low)
            for name, actual, ref, low in zip(("out", "dq", "dk", "dv"), (out, *gradients), *references)
        }
        for parameter, grad in zip((q, k, v), gradients):
            parameter.grad = grad
        torch.optim.SGD((q, k, v), lr=0.001).step()
        if not all(torch.isfinite(item).all().item() for item in (q, k, v)):
            raise AssertionError("Optimizer produced nonfinite parameters")
        statistics.append({"iteration": iteration, "seconds": elapsed, "errors": errors})
        result = {name: item.detach().cpu() for name, item in zip(("out", "dq", "dk", "dv"), (out, *gradients))}
    torch.save(result, output / f"case-{index}.pt")
    report = {
        "backend": backend,
        "layout": layout,
        "causal": causal,
        "heads": [heads, kv_heads, dimension],
        "timings": statistics,
    }
    (output / f"case-{index}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def detect():
    te = importlib.import_module("transformer_engine.pytorch.attention.dot_product_attention.backends")
    megatron = importlib.import_module("megatron.core.transformer.attention")
    if te.flash_attn_func_v4 is None or not megatron.HAVE_FA4:
        raise RuntimeError("Transformer Engine or Megatron did not detect FA4")
    print(json.dumps({"transformer_engine_fa4": te.flash_attn_func_v4.__module__, "megatron_fa4": megatron.HAVE_FA4}))


def run_subprocess(command, log):
    with log.open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=600)


def qualify(output, reference_python):
    run_subprocess([sys.executable, str(HERE / "verify_install.py"), "--runtime"], output / "provenance.json")
    run_subprocess([sys.executable, str(__file__), "--detect"], output / "backend-detection.json")
    for index, case in enumerate(CASES):
        run_subprocess(
            [sys.executable, str(__file__), "--case", str(index), "--output", str(output)], output / f"case-{index}.log"
        )
        if case[0] == "fa4":
            oracle = output / "upstream"
            oracle.mkdir(exist_ok=True)
            run_subprocess(
                [str(reference_python), str(__file__), "--oracle", "--case", str(index), "--output", str(oracle)],
                oracle / f"case-{index}.log",
            )
            torch = importlib.import_module("torch")
            actual = torch.load(output / f"case-{index}.pt", weights_only=True)
            expected = torch.load(oracle / f"case-{index}.pt", weights_only=True)
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    (output / "qualification.json").write_text(
        json.dumps(
            {"status": "passed", "cases": len(CASES), "upstream_comparison": "bitwise; deterministic kernels"}, indent=2
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--reference-python", type=Path)
    parser.add_argument("--case", type=int, choices=range(len(CASES)))
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--detect", action="store_true")
    args = parser.parse_args()
    if args.detect:
        detect()
        return
    if args.output is None:
        parser.error("--output is required")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case is not None:
        run_case(args.case, args.output, args.oracle)
    else:
        if args.reference_python is None:
            parser.error("full qualification requires --reference-python")
        qualify(args.output, args.reference_python)


if __name__ == "__main__":
    main()
