"""GPU test for expert_nvfp4's stochastic rounding (``kernels.nvfp4_quant.stochastic``): weight draws are fixed by
(seed, version, expert), activation draws by the CUDA generator; the round trip, the packed operand and the in-place
write-back agree; and the rounding is unbiased where round-to-nearest is not. Each test runs on the fused sm_100
kernels (hardware ``cvt.rs``) and on the torch reference (``SKYRL_QAT_NVFP4_FUSED=0``).

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

from skyrl.backends.skyrl_train.kernels.nvfp4_quant import fused_rt  # noqa: E402
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
    E2M1_GRID,
    WeightDraw,
    decode_e2m1,
    weight_seed,
)

BLOCK_SCALES = ["absmax", "four_over_six"]
DRAWS = 256


@pytest.fixture(autouse=True, params=["fused", "reference"])
def path(request, monkeypatch):
    monkeypatch.setenv("SKYRL_QAT_NVFP4_FUSED", "1" if request.param == "fused" else "0")
    if request.param == "fused" and not fused_rt.available():
        pytest.skip("fused NVFP4 kernels unavailable")
    return request.param


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


def _scaled(w: torch.Tensor, packed) -> torch.Tensor:
    """``w * gs / sf``: each weight in units of its block's E4M3 scale, on the pack's scales."""
    e, n, k = w.shape
    sf = packed.sf_linear.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1)
    gs = (1.0 / packed.alpha).view(e, 1, 1)
    return torch.where(sf > 0, w.float() * gs / sf, torch.zeros_like(sf))


@pytest.mark.parametrize("block_scale", BLOCK_SCALES)
def test_weight_codes_are_grid_neighbours_on_rn_scales(block_scale):
    """Every stochastic code is one of its scaled value's two E2M1 neighbours (a misrouted random-bit lane or element
    order would not be), on the round-to-nearest scales, one E4M3 step larger only where those would clip."""
    ws = _experts()
    w = torch.stack(ws)
    rn = pack_expert_weights(ws, block_scale, keep_linear=True)
    sr = pack_expert_weights(ws, block_scale, keep_linear=True, draw=_draw(5))
    bump = sr.sf_linear.int() - rn.sf_linear.int()
    assert bump.min() >= 0 and bump.max() <= 1
    vmax = _scaled(w, rn).abs().view(*w.shape[:-1], -1, 16).amax(-1)
    clips = vmax > 6.0
    # A block whose largest value lands exactly on 6 is common (its amax shares the expert amax's bf16 mantissa, so its
    # RN scale is exact) and goes either way on an ulp of the scale arithmetic; every other block must agree.
    decided = (vmax - 6.0).abs() > 1e-5
    assert torch.equal((bump.bool() & decided), (clips & (rn.sf_linear > 0) & decided))
    assert clips.any()
    q = sr.q
    codes = torch.stack([q & 0xF, q >> 4], -1).view(w.shape)
    vals, x = decode_e2m1(codes), _scaled(w, sr)
    grid = torch.tensor(E2M1_GRID, device=w.device)
    lo = grid[(torch.bucketize(x.abs(), grid, right=True) - 1).clamp(0, 7)]
    hi = grid[torch.bucketize(x.abs(), grid, right=True).clamp(max=7)]
    near = lambda a, b: (a - b).abs() <= 1e-3 * b.clamp_min(1)  # noqa: E731 - scale rounding differs by an ulp
    ok = (near(vals.abs(), lo) | near(vals.abs(), hi)) & ((vals == 0) | (torch.sign(vals) == torch.sign(x)))
    assert ok.float().mean().item() > 0.9999
