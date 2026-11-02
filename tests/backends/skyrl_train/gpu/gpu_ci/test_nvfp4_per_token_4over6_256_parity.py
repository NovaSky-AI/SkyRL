"""Bitwise parity of per-token NVFP4 activation quantization, TE (trainer) vs FlashInfer (vLLM sampler), with 4/6 MSE
and the E4M3 bound 256 -- the run-D recipe -- plus the 448 bound as a control.

Trainer: TE 2.19 ``NVFP4Quantizer(row_scaled_nvfp4=True, nvfp4_use_4over6=True, nvfp4_4over6_err_mode="MSE",
nvfp4_e4m3_max=256|448)``, what the TE recipe builds for forward activations under ``NVTE_NVFP4_ROW_SCALED_ACTIVATION=1
NVTE_NVFP4_4OVER6=all NVTE_NVFP4_4OVER6_E4M3_USE_256=all|none``.

Sampler, both MoE quantization points:
* fc1 (GEMM1 input): vLLM's ``quantize_nvfp4_per_token_input`` after SkyRL's worker patch
  (``patch_nvfp4_per_token_e4m3_256``), i.e. ``flashinfer.nvfp4_quantize(per_token_activation=True)`` with the
  patched base scale.
* fc2 (GEMM2 input): the fused TRT-LLM MoE quantizes in-kernel with ``invokeNvfp4QuantAndPerTokenScale`` and
  ``globalScaleInv`` = 1/(256*6) iff ``FLASHINFER_NVFP4_4OVER6=1`` and ``..._E4M3_USE_256=1``
  (``trtllm_fused_moe_runner.cu``). The same kernel is reachable standalone through ``nvfp4_quantize``'s CUDA
  per-token path (``nvfp4_quant_and_per_token_scale_sm100``); it is called with the runner's ``globalScaleInv`` rule
  and an ``expanded_idx_to_permuted_idx`` row map (as the runner passes), linear SF layout.

Asserted per case (T in {1, 7, 128, 1000}, H in {2048, 768}, randn / heavy-tail / zero rows): packed FP4 bytes and
E4M3 block scales equal TE's, and the per-token FP32 scales equal ``1 / fl(fl(1/globalScaleInv) / amax_TE)`` (FlashInfer's
formula on TE's row amax; the value TE's own GEMM derives from that amax is reported, not asserted). The 256 run also
checks that 256 vs 448 changes the bytes (so the test can fail) and that vLLM's constant was patched.

FlashInfer reads its env once per process, so each bound runs in its own subprocess (this file is the worker).
Needs one SM100 GPU; ~1-2 minutes:

    uv run --isolated --extra dev --extra megatron pytest -s \\
        tests/backends/skyrl_train/gpu/gpu_ci/test_nvfp4_per_token_4over6_256_parity.py
"""

import json
import os
import subprocess
import sys

import pytest

SHAPES = [(t, h) for t in (1, 7, 128, 1000) for h in (2048, 768)]


def _env(bound: int) -> dict:
    from skyrl.backends.skyrl_train.weight_sync.nvfp4 import nvfp4_runtime_env

    base = {k: v for k, v in os.environ.items() if not k.startswith(("NVTE_", "FLASHINFER_", "TRTLLM_"))}
    recipe = {
        "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
        "NVTE_NVFP4_4OVER6": "all",
        "NVTE_NVFP4_4OVER6_ERR_MODE": "MSE",
        "NVTE_NVFP4_4OVER6_E4M3_USE_256": "all" if bound == 256 else "none",
    }
    env = {**base, **nvfp4_runtime_env(recipe)}
    assert env["FLASHINFER_NVFP4_4OVER6_E4M3_USE_256"] == ("1" if bound == 256 else "0")
    return env


@pytest.mark.parametrize("bound", [256, 448], ids=["e4m3_256", "e4m3_448_control"])
def test_te_vs_flashinfer_per_token_4over6(bound):
    import torch

    from skyrl.backends.skyrl_train.distributed.megatron.quantization_utils import (
        is_blackwell_or_newer,
    )

    if not torch.cuda.is_available() or not is_blackwell_or_newer():
        pytest.skip("needs an SM100 GPU")
    out = subprocess.run(
        [sys.executable, __file__, str(bound)], env=_env(bound), capture_output=True, text=True, timeout=900
    )
    sys.stderr.write(out.stderr[-6000:])
    lines = [ln for ln in out.stdout.splitlines() if ln.startswith("{")]
    for ln in lines:
        print(ln)
    assert out.returncode == 0, out.stdout[-4000:] + out.stderr[-4000:]
    summary = json.loads(lines[-1])
    assert summary["all_equal"], summary


# ----------------------------------------------------------------------------------------------------- worker


def _cases(torch, t, h):
    g = torch.Generator(device="cuda").manual_seed(1000 * t + h)
    x = torch.randn(t, h, device="cuda", generator=g)
    yield "randn", x.to(torch.bfloat16)
    yield "heavytail", (x * torch.exp(1.5 * torch.randn(t, h, device="cuda", generator=g))).to(torch.bfloat16)
    z = x.to(torch.bfloat16).clone()
    z[0] = 0  # a zero row
    if h >= 32:
        z[-1, :32] = 0  # zero blocks
    yield "zeros", z


def _worker(bound: int) -> int:
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    import flashinfer
    from flashinfer.quantization.fp4_quantization import SfLayout
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer

    from skyrl.backends.skyrl_train.patches.vllm.patch_nvfp4_per_token_e4m3_256 import (
        apply_nvfp4_per_token_e4m3_256_patch,
    )

    applied = apply_nvfp4_per_token_e4m3_256_patch()
    import vllm.model_executor.layers.fused_moe  # noqa: F401  (flashinfer_utils import cycle)
    from vllm.model_executor.layers.quantization.utils import flashinfer_utils

    expected_scale = 1.0 / (bound * 6.0)
    header = {
        "bound": bound,
        "patch_applied": applied,
        "vllm_base_scale": flashinfer_utils.NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE,
        "env": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("FLASHINFER_NVFP4", "NVTE_NVFP4"))},
    }
    print(json.dumps(header), flush=True)
    ok = applied == (bound == 256) and flashinfer_utils.NVFP4_PER_TOKEN_BASE_GLOBAL_SCALE == expected_scale
    # FlashInfer's fused-MoE fc2 rule (trtllm_fused_moe_runner.cu), from the same env.
    fc2_scale_inv = (
        1.0 / (256.0 * 6.0)
        if os.environ.get("FLASHINFER_NVFP4_4OVER6") == "1"
        and os.environ.get("FLASHINFER_NVFP4_4OVER6_E4M3_USE_256") == "1"
        else 1.0 / (448.0 * 6.0)
    )
    ok &= fc2_scale_inv == expected_scale

    def te_quant(x, e4m3_max):
        q = NVFP4Quantizer(
            rowwise=True,
            columnwise=False,
            with_rht=False,
            with_post_rht_amax=False,
            with_2d_quantization=False,
            stochastic_rounding=False,
            row_scaled_nvfp4=True,
            nvfp4_use_4over6=True,
            nvfp4_e4m3_max=e4m3_max,
            nvfp4_4over6_err_mode="MSE",
            with_random_sign_mask=False,
        )
        t_ = q.quantize(x)
        m, k = x.shape
        data = t_._rowwise_data.view(torch.uint8)[:m, : k // 2].contiguous()
        sf = t_._rowwise_scale_inv.view(torch.uint8)[:m, : k // 16].contiguous()
        return data, sf, t_._amax_rowwise.reshape(-1)[:m].float()

    def fi_formula(amax, scale_inv):
        """nvfp4QuantAndPerTokenScaleKernel (4/6, no fast math) on TE's row amax, in IEEE fp32 (torch div is _rn)."""
        one = torch.ones((), dtype=torch.float32, device="cuda")
        global_scale = one / torch.tensor(scale_inv, dtype=torch.float32, device="cuda")
        enc = torch.clamp(global_scale / amax, max=torch.finfo(torch.float32).max)
        enc = torch.where(enc == 0, one, enc)
        dec = one / enc
        return torch.where(amax == 0, torch.zeros_like(amax), dec), global_scale.item()

    def bits_equal(a, b):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.dtype == torch.float32:
            a, b = a.view(torch.int32), b.view(torch.int32)
        return bool(torch.equal(a, b))

    sensitive = False
    for t, h in SHAPES:
        for name, x in _cases(torch, t, h):
            te_data, te_sf, te_amax = te_quant(x, bound)
            ref_tok, global_scale = fi_formula(te_amax, expected_scale)
            rec = {"case": f"{name}_{t}x{h}", "global_scale": global_scale}

            # fc1: vLLM's own entry point, with the (patched) module constant.
            d1, s1, tok1 = flashinfer_utils.quantize_nvfp4_per_token_input(x)
            d1 = d1.view(torch.uint8).reshape(t, h // 2)
            s1 = s1.view(torch.uint8).reshape(t, -1)[:, : h // 16]
            tok1 = tok1.reshape(-1).float()
            rec["fc1"] = {
                "data": bits_equal(d1, te_data),
                "sf": bits_equal(s1, te_sf),
                "tok": bits_equal(tok1, ref_tok),
                "tok_vs_amax_div": bits_equal(tok1, te_amax / (bound * 6.0)),  # informational
            }

            # fc2-style: the fused runner's kernel, its globalScaleInv rule, and a row map (some rows skipped).
            perm = torch.randperm(t, device="cuda", generator=torch.Generator(device="cuda").manual_seed(t))
            row_map = perm.to(torch.int32)
            if t > 2:
                row_map[t // 2] = -1
            covered = row_map[row_map >= 0].long()
            d2, s2, tok2 = flashinfer.nvfp4_quantize(
                x,
                fc2_scale_inv,
                sfLayout=SfLayout.layout_linear,
                per_token_activation=True,
                expanded_idx_to_permuted_idx=row_map,
            )
            d2 = d2.view(torch.uint8).reshape(t, h // 2)[covered]
            s2 = s2.view(torch.uint8).reshape(t, -1)[:, : h // 16][covered]
            tok2 = tok2.reshape(-1).float()[covered]
            rec["fc2"] = {
                "data": bits_equal(d2, te_data[covered]),
                "sf": bits_equal(s2, te_sf[covered]),
                "tok": bits_equal(tok2, ref_tok[covered]),
            }
            case_ok = all(rec["fc1"][k] for k in ("data", "sf", "tok")) and all(rec["fc2"].values())
            if not case_ok:
                rec["data_diff"] = int((d1 != te_data).sum())
                rec["sf_diff"] = int((s1 != te_sf).sum())
            ok &= case_ok
            if bound == 256 and name == "randn" and t >= 128:
                other_data, other_sf, _ = te_quant(x, 448)
                sensitive |= not (torch.equal(other_data, te_data) and torch.equal(other_sf, te_sf))
            print(json.dumps(rec), flush=True)
    if bound == 256:
        ok &= sensitive
    print(json.dumps({"bound": bound, "all_equal": bool(ok), "bound_changes_bytes": sensitive}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_worker(int(sys.argv[1])))
