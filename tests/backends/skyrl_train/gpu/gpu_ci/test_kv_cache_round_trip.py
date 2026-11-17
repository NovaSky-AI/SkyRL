"""GPU tests: the one-pass KV cache round trip is bit-identical to vLLM's own cache writer + reader (sm_100).

Run with:
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/test_kv_cache_round_trip.py

Against a live engine (GLM-5.3 layer 0, vLLM 0.32 FlashMLA sparse) the reference itself matched the cache bytes the
engine wrote, bit for bit, for both formats; that needs a DSA checkpoint and is not run here.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="needs sm_100 (B200/B300)"
)
pytest.importorskip("vllm")

from skyrl.backends.skyrl_train.kernels.kv_cache_round_trip import (  # noqa: E402
    kv_cache_round_trip,
)
from skyrl.backends.skyrl_train.workers.megatron.quantization import (  # noqa: E402
    fake_kv_cache_qat as kvq,
)

THETA = 8_000_000.0  # GLM-5.3


def _rows(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    kv = torch.randn(n, 576, device="cuda", generator=g) * torch.exp(torch.randn(576, device="cuda", generator=g))
    kv = kv.bfloat16()
    kv[::97, :16] = 0  # all-zero blocks: the 2^-9 scale floor
    kv[1::89, :32] = -0.0  # exact negative zeros encode as +0
    kv[2::83, 100] = 1e4  # one outlier per block: the rest round to (signed) zero
    return kv


@pytest.mark.parametrize("fmt", list(kvq.KvCacheFormat))
def test_one_pass_is_bit_identical_to_vllm_writer_and_reader(fmt):
    n = 50_000
    kv = _rows(n, 0)
    pos = torch.randint(0, kvq.MAX_POSITIONS, (n,), device="cuda")
    w = (torch.rand(512, device="cuda") + 0.3).bfloat16()
    ref = kvq.to_megatron_key_order(kvq.cache_round_trip(kv, pos, w, 1e-5, THETA, fmt))
    got = kv_cache_round_trip(
        kv, pos, w, 1e-5, kvq._cos_sin_cache(kv.device, THETA), fmt == kvq.KvCacheFormat.NVFP4_DS_MLA
    )
    assert torch.equal(got.view(torch.int16), ref.view(torch.int16))


def test_strided_input_rows():
    """The kernel reads the ``linear_kv_down_proj`` output in place, at any row stride."""
    base = _rows(4_096, 1)
    wide = torch.zeros(4_096, 640, dtype=torch.bfloat16, device="cuda")
    wide[:, :576] = base
    pos = torch.arange(4_096, device="cuda")
    w = torch.ones(512, dtype=torch.bfloat16, device="cuda")
    cs = kvq._cos_sin_cache(base.device, THETA)
    assert torch.equal(kv_cache_round_trip(wide[:, :576], pos, w, 1e-5, cs, True),
                       kv_cache_round_trip(base, pos, w, 1e-5, cs, True))  # fmt: skip


def test_round_trip_error_is_the_format_s():
    """Sanity: NVFP4 rounding error ~9% of the latent, E4M3 RoPE ~3%; FP8 latent ~2-3%, bf16 RoPE ~0.3%."""
    n = 8_192
    kv = torch.randn(n, 576, device="cuda").bfloat16()
    pos = torch.zeros(n, dtype=torch.int64, device="cuda")  # position 0: RoPE is the identity
    w = torch.ones(512, dtype=torch.bfloat16, device="cuda")
    cs = kvq._cos_sin_cache(kv.device, THETA)
    x = kv.float()
    normed = x[:, :512] * torch.rsqrt(x[:, :512].pow(2).mean(-1, keepdim=True) + 1e-5)
    rope = kvq.to_megatron_key_order(torch.cat([normed, x[:, 512:]], -1))[:, 512:]
    rel = lambda a, b: ((a - b).norm() / b.norm()).item()  # noqa: E731
    out4 = kv_cache_round_trip(kv, pos, w, 1e-5, cs, True).float()
    out8 = kv_cache_round_trip(kv, pos, w, 1e-5, cs, False).float()
    assert 0.06 < rel(out4[:, :512], normed) < 0.12 and 0.015 < rel(out4[:, 512:], rope) < 0.04
    assert 0.01 < rel(out8[:, :512], normed) < 0.04 and rel(out8[:, 512:], rope) < 0.005


def _unquantized(kv: torch.Tensor) -> torch.Tensor:
    """fp32 cache-row values at position 0 (RoPE is the identity) with a unit norm weight, megatron's column order."""
    x = kv.float()
    normed = x[:, :512] * torch.rsqrt(x[:, :512].pow(2).mean(-1, keepdim=True) + 1e-5)
    return kvq.to_megatron_key_order(torch.cat([normed, x[:, 512:]], -1))


def _inputs(n=2_048):
    kv = (torch.randn(n, 576, device="cuda") * 3).bfloat16()
    pos = torch.zeros(n, dtype=torch.int64, device="cuda")
    w = torch.ones(512, dtype=torch.bfloat16, device="cuda")
    return kv, pos, w, kvq._cos_sin_cache(kv.device, THETA)


def test_four_over_six_never_reconstructs_a_block_worse():
    kv, pos, w, cs = _inputs()
    ref = _unquantized(kv)[:, :512].view(-1, 32, 16)
    err = lambda out: (out.float()[:, :512].view(-1, 32, 16) - ref).pow(2).sum(-1)  # noqa: E731
    e6 = err(kv_cache_round_trip(kv, pos, w, 1e-5, cs, True))
    e46 = err(kv_cache_round_trip(kv, pos, w, 1e-5, cs, True, four_over_six=True))
    # The kernel compares its own fp32 values (vLLM's RMSNorm); these differ from torch's by an ulp, so a block whose two
    # candidates tie can land either way.
    assert (e46 > e6 * (1 + 1e-3) + 1e-9).float().mean() < 1e-3 and (e46 < e6).float().mean() > 0.2
    assert torch.equal(
        kv_cache_round_trip(kv, pos, w, 1e-5, cs, True, four_over_six=True)[:, 512:],
        kv_cache_round_trip(kv, pos, w, 1e-5, cs, True)[:, 512:],
    )


def _seed(v: int) -> torch.Tensor:
    return torch.tensor([v], dtype=torch.int64, device="cuda")


@pytest.mark.parametrize("hw_sr", [True, False])
@pytest.mark.parametrize("nvfp4, four_over_six", [(True, False), (True, True), (False, False)])
def test_stochastic_rows_are_seeded_and_unbiased(hw_sr, nvfp4, four_over_six):
    if hw_sr and torch.cuda.get_device_capability() not in ((10, 0), (10, 3)):
        pytest.skip("hardware stochastic rounding needs sm_100 / sm_103")
    kv, pos, w, cs = _inputs()
    run = lambda seed: kv_cache_round_trip(  # noqa: E731
        kv, pos, w, 1e-5, cs, nvfp4, four_over_six, seed=seed, hw_sr=hw_sr
    ).float()
    a = run(_seed(3))
    assert torch.equal(a, run(_seed(3))) and not torch.equal(a, run(_seed(4)))
    ref, rn = _unquantized(kv), kv_cache_round_trip(kv, pos, w, 1e-5, cs, nvfp4, four_over_six).float()
    draws = 256
    mean = sum(run(_seed(100 + i)) for i in range(draws)) / draws
    bias = lambda x, cols: (x[:, cols] - ref[:, cols]).abs().mean().item()  # noqa: E731
    parts = [slice(0, 512), slice(512, 576)] if nvfp4 else [slice(0, 512)]  # fp8_ds_mla keeps bf16 RoPE
    for cols in parts:
        assert bias(mean, cols) < 0.25 * bias(rn, cols), cols
    if not nvfp4:
        assert torch.equal(a[:, 512:], rn[:, 512:])
