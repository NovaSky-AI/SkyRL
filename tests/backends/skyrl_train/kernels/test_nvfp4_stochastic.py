"""Torch reference stochastic rounding of NVFP4 values (``kernels.nvfp4_quant.stochastic``), on CPU."""

import pytest
import torch

from skyrl.backends.skyrl_train.kernels.nvfp4_quant import stochastic as sr

GRID = torch.tensor(sr.E2M1_GRID)


def _values(codes: torch.Tensor) -> torch.Tensor:
    return sr.decode_e2m1(codes)


def test_rounds_to_a_neighbour_and_keeps_exact_values():
    v = torch.linspace(-6.0, 6.0, 4097)
    q = _values(sr.stochastic_e2m1(v, torch.rand(v.shape)))
    a, qa = v.abs(), q.abs()
    lo = GRID[(torch.bucketize(a, GRID, right=True) - 1).clamp(0, 7)]
    hi = GRID[torch.bucketize(a, GRID, right=True).clamp(max=7)]
    assert torch.all((qa == lo) | (qa == hi))
    assert torch.all(torch.sign(q) * torch.sign(v) >= 0)
    exact = torch.cat([GRID, -GRID[1:]])
    for u in (torch.zeros(exact.shape), torch.full(exact.shape, 0.999999)):
        assert torch.equal(_values(sr.stochastic_e2m1(exact, u)), exact)


def test_is_unbiased():
    torch.manual_seed(0)
    v = torch.tensor([0.1, 0.3, 0.7, 1.2, 1.9, 2.5, 3.3, 4.9, 5.99, -0.4, -2.2, -5.1])
    n = 200_000
    q = _values(sr.stochastic_e2m1(v.expand(n, -1), torch.rand(n, v.numel())))
    assert torch.allclose(q.mean(0), v, atol=0.02)


def test_saturates_above_six():
    assert torch.equal(
        _values(sr.stochastic_e2m1(torch.tensor([7.0, -9.0]), torch.zeros(2))), torch.tensor([6.0, -6.0])
    )


def test_pack_nibbles_low_nibble_is_the_even_element():
    codes = torch.tensor([[1, 2, 15, 8]], dtype=torch.uint8)
    assert sr.pack_nibbles(codes).tolist() == [[0x21, 0x8F]]


def test_requantize_raises_clipping_scales_one_step():
    """A block whose largest value would land above 6 on its scale gets the next E4M3 scale; others keep theirs."""
    x = torch.zeros(2, 16)
    x[0, 0], x[1, 0] = 6.5, 5.5
    sf = torch.full((2, 1), 1.0).to(torch.float8_e4m3fn).view(torch.uint8)
    q, sf2 = sr.stochastic_requantize(x, torch.ones(2), sf, torch.rand(2, 16))
    assert sf2[0, 0] == sf[0, 0] + 1 and sf2[1, 0] == sf[1, 0]
    scale0 = sr.e4m3_bytes_to_float(sf2[0])
    assert 6.5 / scale0 <= 6.0
    assert q.shape == (2, 8)


def test_requantize_zero_scale_blocks_give_zero():
    x = torch.randn(1, 32)
    sf = torch.tensor([[0, 0x38]], dtype=torch.uint8)
    q, _ = sr.stochastic_requantize(x, torch.ones(1), sf, torch.rand(1, 32))
    assert torch.all((q[0, :8] & 0x77) == 0)


def test_expert_uniforms_depend_only_on_seed_and_stream():
    draw = sr.WeightDraw(seed=sr.weight_seed(42, 3), streams=(11, 12, 13))
    u = sr.expert_uniforms((3, 4, 32), draw, False, torch.device("cpu"))
    again = sr.expert_uniforms((3, 4, 32), draw, False, torch.device("cpu"))
    assert torch.equal(u, again)
    # Memory order reversed (Megatron's buffer layout): the same expert gets the same uniforms.
    rev = sr.expert_uniforms((3, 4, 32), draw, True, torch.device("cpu"))
    assert torch.equal(rev, u.flip(0))
    other = sr.expert_uniforms((3, 4, 32), sr.WeightDraw(sr.weight_seed(42, 4), draw.streams), False, "cpu")
    assert not torch.equal(u, other)
    with pytest.raises(ValueError, match="streams"):
        sr.expert_uniforms((2, 4, 32), draw, False, torch.device("cpu"))


def test_seeds_and_streams_are_distinct():
    streams = {sr.expert_stream(layer, fc2, e) for layer in range(1, 80) for fc2 in (False, True) for e in range(256)}
    assert len(streams) == 79 * 2 * 256
    assert len({sr.weight_seed(1, v) for v in range(1000)}) == 1000
    assert sr.weight_seed(1, 0) != sr.weight_seed(2, 0)


def test_weight_draw_follows_the_registry_and_version(monkeypatch):
    from skyrl.backends.skyrl_train.workers.megatron.quantization import expert_nvfp4

    w0, w1 = torch.nn.Parameter(torch.zeros(1)), torch.nn.Parameter(torch.zeros(1))
    monkeypatch.setattr(expert_nvfp4, "_WEIGHT_STREAMS", {id(w0): 5, id(w1): 6})
    monkeypatch.setattr(expert_nvfp4, "_WEIGHT_ROUNDING", expert_nvfp4._WeightRounding())
    assert expert_nvfp4.weight_draw([w0, w1]) is None
    expert_nvfp4.set_weight_rounding(True, base_seed=7)
    d0 = expert_nvfp4.weight_draw([w0, w1])
    assert d0 == sr.WeightDraw(sr.weight_seed(7, 0), (5, 6))
    assert expert_nvfp4.weight_draw([w0, w1]) == d0  # fprop, dgrad and recompute of a version agree
    expert_nvfp4.advance_weight_rounding()
    assert expert_nvfp4.weight_draw([w0, w1]).seed == sr.weight_seed(7, 1)
    with pytest.raises(RuntimeError, match="not registered"):
        expert_nvfp4.weight_draw([torch.nn.Parameter(torch.zeros(1))])
