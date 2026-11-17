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
