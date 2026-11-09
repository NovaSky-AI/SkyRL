"""GPU test for the NVFP4 expert quantizer (``kernels.nvfp4_quant.quantize_experts``, FlashInfer's
``fp4_quantize``): absmax matches vLLM's own quantizer byte for byte, including zero-scale blocks, and 4/6
picks other block scales with no more error, while the process env cannot change the weight grid.

uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_nvfp4_flashinfer_quant.py
"""

import os

import pytest
import torch

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10, reason="needs Blackwell"
    ),
]

pytest.importorskip("flashinfer")
pytest.importorskip("vllm")

from skyrl.backends.skyrl_train.kernels.nvfp4_quant import (  # noqa: E402
    flashinfer_nvfp4_env,
    quantize_experts,
)


def _weights(experts=16, n=512, k=1024, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(experts, n, k, device="cuda", generator=g) * 0.02
    w *= torch.exp(torch.randn(experts, n, 1, device="cuda", generator=g))  # rows of very different scale
    w[0, :, :16] = 0.0  # zero blocks
    w[1, ::2, :16] = -0.0
    w[2] *= 1e-6  # under the expert's own global scale, but small blocks next to w[2, 0, 0]
    w[2, 0, 0] = 1.0
    return w.to(torch.bfloat16)


def _block_err(q, sf, gs, w):
    """Squared error per 16-block of the decoded ``(q, sf)``."""
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device=q.device)
    codes = torch.stack([q & 0xF, q >> 4], -1).flatten(-2).long()
    dq = lut[codes].view(*w.shape[:-1], -1, 16) * (sf.float() / gs.view(-1, 1, 1)).unsqueeze(-1)
    return (dq.flatten(-2) - w.float()).pow(2).view(-1, 16).sum(-1)


def test_absmax_matches_vllm_and_four_over_six_has_no_more_error(monkeypatch):
    import vllm.model_executor.layers.quantization.online.nvfp4 as online_nvfp4

    w = _weights()
    # vLLM's own online quantizer (scaled_fp4_quant per expert) and its global scale.
    q_vllm, sf_vllm, scale_2 = online_nvfp4._quantize_moe_weight_to_nvfp4(w)
    gs = 1.0 / scale_2
    # The engines' activation env, and a stray exact-math setting, must not leak into the weight call.
    stray = {**flashinfer_nvfp4_env(True), "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1"}
    for key, value in stray.items():
        monkeypatch.setenv(key, value)

    q, sf = quantize_experts(w, gs, "absmax")
    assert torch.equal(q, q_vllm) and torch.equal(sf.view(torch.uint8), sf_vllm.view(torch.uint8))

    q46, sf46 = quantize_experts(w, gs, "four_over_six")
    assert not torch.equal(sf46.view(torch.uint8), sf.view(torch.uint8))
    err, err46 = _block_err(q, sf, gs, w), _block_err(q46, sf46, gs, w)
    # FlashInfer compares the candidates on its fast error path, so near-ties may go either way.
    assert (err46 > err * (1 + 1e-2) + 1e-12).float().mean() < 1e-3
    assert err46.sum() < err.sum()
    # One launch for all experts equals one per expert.
    for e in (0, 5):
        qe, sfe = quantize_experts(w[e : e + 1], gs[e : e + 1], "four_over_six")
        assert torch.equal(qe[0], q46[e]) and torch.equal(sfe[0].view(torch.uint8), sf46[e].view(torch.uint8))
    assert all(os.environ[k] == v for k, v in stray.items())  # restored
