"""The torch fallback for apex's ``fused_weight_gradient_mlp_cuda`` and the DDP config fields.

uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/patches/megatron/test_grad_accum_fusion_fallback.py
"""

import dataclasses

import pytest
import torch

from skyrl.backends.skyrl_train.patches.megatron import (
    patch_grad_accum_fusion_fallback as fallback,
)


def _inputs(dtype):
    torch.manual_seed(0)
    total_input = torch.randn(2, 5, 16, dtype=dtype)  # [s, b, in]
    grad_output = torch.randn(2, 5, 8, dtype=dtype)  # [s, b, out]
    return total_input, grad_output


@pytest.mark.parametrize(
    "device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA"))]
)
def test_wgrad_gemm_accum_fp32_accumulates_into_main_grad(device):
    total_input, grad_output = (t.to(device) for t in _inputs(torch.bfloat16))
    main_grad = torch.randn(8, 16, dtype=torch.float32, device=device)
    expected = main_grad + grad_output.reshape(-1, 8).float().t() @ total_input.reshape(-1, 16).float()
    fallback.wgrad_gemm_accum_fp32(total_input, grad_output, main_grad)
    torch.testing.assert_close(main_grad, expected, rtol=1e-2, atol=1e-2)


def test_wgrad_gemm_accum_fp16_accumulates_into_main_grad():
    total_input, grad_output = _inputs(torch.float32)
    main_grad = torch.randn(8, 16, dtype=torch.float32)
    expected = main_grad + grad_output.reshape(-1, 8).t() @ total_input.reshape(-1, 16)
    fallback.wgrad_gemm_accum_fp16(total_input, grad_output, main_grad)
    torch.testing.assert_close(main_grad, expected)


def test_patch_installs_the_fallback_without_apex(monkeypatch):
    layers = pytest.importorskip("megatron.core.tensor_parallel.layers", reason="requires the megatron extra")
    try:
        import fused_weight_gradient_mlp_cuda  # noqa: F401

        pytest.skip("apex's extension is installed; the fallback is not used")
    except ImportError:
        pass
    monkeypatch.setattr(fallback, "_APPLIED", False)
    monkeypatch.setattr(layers, "_grad_accum_fusion_available", False, raising=False)
    monkeypatch.setattr(layers, "fused_weight_gradient_mlp_cuda", None, raising=False)
    assert fallback.apply_grad_accum_fusion_fallback_patch()
    assert layers._grad_accum_fusion_available
    assert layers.fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32 is fallback.wgrad_gemm_accum_fp32


def test_ddp_config_fields_exist_in_megatron():
    """Every MegatronDDPConfig field (align_param_gather included) is set on megatron-core's
    DistributedDataParallelConfig by the worker, so each must be a field there."""
    ddp = pytest.importorskip("megatron.core.distributed", reason="requires the megatron extra")
    from skyrl.train.config.config import MegatronDDPConfig

    ours = {f.name for f in dataclasses.fields(MegatronDDPConfig)}
    theirs = {f.name for f in dataclasses.fields(ddp.DistributedDataParallelConfig)}
    assert "align_param_gather" in ours
    assert ours <= theirs, sorted(ours - theirs)
