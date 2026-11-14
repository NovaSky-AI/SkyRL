"""GPU test for the NVFP4 expert quantizer (``kernels.nvfp4_quant.quantize_experts``, FlashInfer's
``fp4_quantize``): absmax matches vLLM's own quantizer byte for byte, including zero-scale blocks, and 4/6
picks other block scales with no more error, while the process env cannot change the weight grid; the ModelOpt
sync's bytes decode to the trainer's round trip.

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


@pytest.mark.parametrize("block_scale", ["absmax", "four_over_six"])
def test_modelopt_sync_bytes_decode_to_the_trainers_round_trip(block_scale):
    """expert_nvfp4's ModelOpt sync ships, per expert, exactly the NVFP4 the trainer rounds to: the serializer's
    ``quantize_nvfp4_matrix`` (vLLM's global scale over the whole ``[gate; up]`` matrix) decodes to the trainer's
    ``fake_nvfp4_quantize_weights`` bit for bit, and its codes / scales / decode scale are the trainer's packed
    operand's (the bytes vLLM loads as-is)."""
    from skyrl.backends.skyrl_train.kernels.nvfp4_quant import fake_experts
    from skyrl.backends.skyrl_train.weight_sync import nvfp4
    from skyrl.backends.skyrl_train.workers.megatron.quantization import expert_nvfp4

    w = _weights(experts=8, n=512, k=1024, seed=1)
    w[0] = w[0] * 0 + 0.25  # constant expert
    experts = list(w.unbind(0))
    trainer = expert_nvfp4.fake_nvfp4_quantize_weights(experts, block_scale)
    packed = fake_experts.pack_expert_weights(experts, block_scale, keep_linear=True)
    settings = nvfp4.Nvfp4WeightQuantSettings(block_scale)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
    for e, (we, te) in enumerate(zip(experts, trainer)):
        q, sf, decode = nvfp4.quantize_nvfp4_matrix(we, settings)
        codes = torch.stack([lut[(q & 0xF).long()], lut[(q >> 4).long()]], -1).flatten(-2)
        y = (codes.view(*q.shape[:-1], -1, 16) * sf.float().unsqueeze(-1)).flatten(-2) * decode
        assert torch.equal(y.bfloat16().view(torch.int16), te.view(torch.int16)), e
        i = packed.gemm_expert_index(torch.tensor(e)).item()
        nonzero = sf.view(torch.uint8) != 0  # zero-scale blocks decode to 0 either way (sign-masked on the wire)
        assert torch.equal(sf.view(torch.uint8), packed.sf_linear[i]), e
        assert torch.equal(q.view(512, -1, 8)[nonzero], packed.q[i].view(512, -1, 8)[nonzero]), e
        assert decode.item() == packed.alpha[i].item(), e
