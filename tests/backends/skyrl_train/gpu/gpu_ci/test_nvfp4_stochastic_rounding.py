"""GPU test for expert_nvfp4's stochastic rounding (``kernels.nvfp4_quant.stochastic``): weight draws are fixed by
(seed, version, expert), activation draws by the CUDA generator; the round trip, the packed operand and the in-place
write-back agree; and the rounding is unbiased where round-to-nearest is not.

uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_nvfp4_stochastic_rounding.py
"""

import pytest
import torch

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="needs sm_100"
    ),
]

pytest.importorskip("flashinfer")

from skyrl.backends.skyrl_train.kernels.nvfp4_quant.fake_experts import (  # noqa: E402
    fake_quantize_expert_weights,
    pack_expert_weights,
)
from skyrl.backends.skyrl_train.kernels.nvfp4_quant.fp4_grouped_gemm import (  # noqa: E402
    tile_layout,
)
from skyrl.backends.skyrl_train.kernels.nvfp4_quant.per_token import (  # noqa: E402
    fake_quantize_per_token,
    pack_per_token,
)
from skyrl.backends.skyrl_train.kernels.nvfp4_quant.stochastic import (  # noqa: E402
    WeightDraw,
    weight_seed,
)

BLOCK_SCALES = ["absmax", "four_over_six"]
DRAWS = 256


def _experts(e=4, n=256, k=512, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(e, n, k, device="cuda", generator=g) * 0.02
    w *= torch.exp(torch.randn(e, n, 1, device="cuda", generator=g))
    return list(w.to(torch.bfloat16).unbind(0))


def _draw(version=0, e=4):
    return WeightDraw(weight_seed(1234, version), tuple(range(100, 100 + e)))


def _bias(mean: torch.Tensor, x: torch.Tensor) -> float:
    return (mean - x.float()).abs().mean().item()


@pytest.mark.parametrize("block_scale", BLOCK_SCALES)
def test_weight_draw_is_fixed_by_seed_version_and_expert(block_scale):
    ws = _experts()
    a = fake_quantize_expert_weights(ws, block_scale, _draw(0))
    assert all(torch.equal(x, y) for x, y in zip(a, fake_quantize_expert_weights(ws, block_scale, _draw(0))))
    assert not all(torch.equal(x, y) for x, y in zip(a, fake_quantize_expert_weights(ws, block_scale, _draw(1))))
    assert not all(torch.equal(x, y) for x, y in zip(a, fake_quantize_expert_weights(ws, block_scale)))
    # An expert's draw follows its stream, not its position in the grouped GEMM.
    sub = fake_quantize_expert_weights(ws[2:], block_scale, WeightDraw(_draw(0).seed, _draw(0).streams[2:]))
    assert torch.equal(sub[0], a[2]) and torch.equal(sub[1], a[3])


@pytest.mark.parametrize("block_scale", BLOCK_SCALES)
def test_weight_pack_matches_the_round_trip(block_scale):
    ws = _experts()
    packed = pack_expert_weights(ws, block_scale, keep_linear=True, draw=_draw(3))
    rt = fake_quantize_expert_weights(ws, block_scale, _draw(3))
    assert all(torch.equal(x, y) for x, y in zip(packed.dequantize(), rt))


@pytest.mark.parametrize("block_scale", BLOCK_SCALES)
def test_weight_rounding_is_unbiased(block_scale):
    ws = _experts(e=2, n=128, k=256)
    w = torch.stack(ws)
    rn = torch.stack(fake_quantize_expert_weights(ws, block_scale)).float()
    mean = sum(torch.stack(fake_quantize_expert_weights(ws, block_scale, _draw(v, 2))).float() for v in range(DRAWS))
    mean /= DRAWS
    assert _bias(mean, w) < 0.25 * _bias(rn, w)


@pytest.mark.parametrize("four_over_six", [False, True])
def test_activation_draw_follows_the_cuda_generator(four_over_six):
    x = (torch.randn(384, 2048, device="cuda") * torch.rand(384, 1, device="cuda") * 4).to(torch.bfloat16)
    torch.cuda.manual_seed(5)
    a = fake_quantize_per_token(x, four_over_six=four_over_six, stochastic=True)
    torch.cuda.manual_seed(5)
    b = fake_quantize_per_token(x, four_over_six=four_over_six, stochastic=True)
    assert torch.equal(a, b)
    assert not torch.equal(a, fake_quantize_per_token(x, four_over_six=four_over_six, stochastic=True))
    assert not torch.equal(a, fake_quantize_per_token(x, four_over_six=four_over_six))


@pytest.mark.parametrize("four_over_six", [False, True])
def test_activation_pack_write_back_matches_the_round_trip(four_over_six):
    m_splits = [100, 0, 156, 128]
    x = (torch.randn(sum(m_splits), 2048, device="cuda") * 3).to(torch.bfloat16)
    layout = tile_layout(m_splits, x.device)
    torch.cuda.manual_seed(7)
    rt = fake_quantize_per_token(x, four_over_six=four_over_six, stochastic=True)
    torch.cuda.manual_seed(7)
    xc = x.clone()
    pack_per_token(xc, layout.dst, layout.padded_rows, four_over_six, write_back=True, stochastic=True)
    assert torch.equal(xc, rt)


@pytest.mark.parametrize("four_over_six", [False, True])
def test_activation_rounding_is_unbiased(four_over_six):
    x = (torch.randn(64, 1024, device="cuda") * 2).to(torch.bfloat16)
    rn = fake_quantize_per_token(x, four_over_six=four_over_six).float()
    mean = sum(fake_quantize_per_token(x, four_over_six=four_over_six, stochastic=True).float() for _ in range(DRAWS))
    mean /= DRAWS
    assert _bias(mean, x) < 0.25 * _bias(rn, x)
