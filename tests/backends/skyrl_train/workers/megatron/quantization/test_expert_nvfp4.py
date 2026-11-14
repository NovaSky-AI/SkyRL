"""CPU tests for expert_nvfp4's grouped-GEMM hook, its input rounding, its BF16-layer skip and its install.

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/workers/megatron/quantization/test_expert_nvfp4.py

The NVFP4 quantizer needs Blackwell; here a stand-in rounding exercises the dispatch. The GPU test
``tests/backends/skyrl_train/gpu/gpu_ci/megatron/test_expert_nvfp4.py`` covers the real one. Config validation is
in ``tests/train/test_config.py``.
"""

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.skyrl_train.workers.megatron.quantization import expert_nvfp4


def _round(w: torch.Tensor) -> torch.Tensor:
    """Stand-in for the NVFP4 round trip: visibly changes every weight."""
    return torch.round(w * 2) / 2


def _round_all(weights: list[torch.Tensor]) -> list[torch.Tensor]:
    return [_round(w) for w in weights]


def _gemm(A, B, out, *, layout="TN"):
    """The ``general_grouped_gemm`` contract TE's legacy GroupedLinear path uses, for 2 GEMMs:
    fprop ``TN`` (out_i = B_i @ A_i^T), dgrad ``NN`` (out_i = B_i @ A_i), wgrad ``NT`` (out_i = B_i^T @ A_i)."""
    for i, (a, b) in enumerate(zip(A, B)):
        if layout == "TN":
            out[i] = b @ a.t()
        elif layout == "NN":
            out[i] = b @ a
        else:
            out[i] = b.t() @ a


_this_module = sys.modules[__name__]
general_grouped_gemm = _gemm


class _ToyGroupedLinear(torch.autograd.Function):
    """TE ``_GroupedLinear``'s data flow: weights are saved for backward, every GEMM goes through the
    module-level ``general_grouped_gemm``, and wgrad never reads the weights."""

    @staticmethod
    def forward(ctx, x0, x1, w0, w1):
        out = [None, None]
        _this_module.general_grouped_gemm([w0, w1], [x0, x1], out, layout="TN")
        ctx.save_for_backward(x0, x1, w0, w1)
        return out[0], out[1]

    @staticmethod
    def backward(ctx, g0, g1):
        x0, x1, w0, w1 = ctx.saved_tensors
        dgrad, wgrad = [None, None], [None, None]
        _this_module.general_grouped_gemm([w0, w1], [g0, g1], dgrad, layout="NN")
        _this_module.general_grouped_gemm([x0, x1], [g0, g1], wgrad, layout="NT")
        return dgrad[0], dgrad[1], wgrad[0], wgrad[1]


@pytest.fixture
def hooked(monkeypatch):
    monkeypatch.setattr(expert_nvfp4, "_ROUTED_EXPERT_WEIGHTS", {})
    monkeypatch.setattr(_this_module, "general_grouped_gemm", expert_nvfp4.wrap_grouped_gemm(_gemm, _round_all))


def _register(*weights):
    for w in weights:
        expert_nvfp4._ROUTED_EXPERT_WEIGHTS[id(w)] = w


def _toy_case():
    torch.manual_seed(0)
    w = [torch.nn.Parameter(torch.randn(4, 8)) for _ in range(2)]
    x = [torch.randn(3, 8, requires_grad=True) for _ in range(2)]
    g = [torch.randn(3, 4) for _ in range(2)]
    return w, x, g


def test_registered_weights_are_rounded_in_fprop_and_dgrad_only(hooked):
    """Output and dgrad use the rounded weights; wgrad is the unrounded straight-through gradient."""
    w, x, g = _toy_case()
    _register(*w)
    outs = _ToyGroupedLinear.apply(x[0], x[1], w[0], w[1])
    torch.autograd.backward(outs, g)
    for i in range(2):
        wq = _round(w[i].detach())
        assert not torch.equal(wq, w[i].detach())
        torch.testing.assert_close(outs[i], x[i].detach() @ wq.t(), rtol=0, atol=0)
        torch.testing.assert_close(x[i].grad, g[i] @ wq, rtol=0, atol=0)
        torch.testing.assert_close(w[i].grad, g[i].t() @ x[i].detach(), rtol=0, atol=0)


def test_unregistered_weights_pass_through(hooked):
    """Other grouped linears (and every weight before registration) compute on the raw weights."""
    w, x, g = _toy_case()
    outs = _ToyGroupedLinear.apply(x[0], x[1], w[0], w[1])
    torch.autograd.backward(outs, g)
    for i in range(2):
        torch.testing.assert_close(outs[i], x[i].detach() @ w[i].detach().t(), rtol=0, atol=0)
        torch.testing.assert_close(x[i].grad, g[i] @ w[i].detach(), rtol=0, atol=0)


def test_partially_registered_group_raises(hooked):
    w, x, _ = _toy_case()
    _register(w[0])
    with pytest.raises(RuntimeError, match="1 of 2 registered"):
        _ToyGroupedLinear.apply(x[0], x[1], w[0], w[1])


def test_lookalike_tensor_is_not_registered(hooked):
    """Registration is by object: a copy of a registered weight (e.g. a dtype cast) is not rounded."""
    w, _, _ = _toy_case()
    _register(*w)
    assert expert_nvfp4._is_registered(w[0])
    assert not expert_nvfp4._is_registered(w[0].detach().clone())


def _round_input(x: torch.Tensor) -> torch.Tensor:
    """Stand-in for the activation round trip, distinct from ``_round``."""
    return torch.round(x * 4) / 4


class _ToyGroupedLinearModule(torch.nn.Module):
    """``TEGroupedLinear``'s call shape: ``forward(x, m_splits)`` over two experts."""

    def __init__(self, w0, w1):
        super().__init__()
        self.w0, self.w1 = w0, w1

    def forward(self, x, m_splits):
        x0, x1 = torch.split(x, m_splits)
        return torch.cat(_ToyGroupedLinear.apply(x0, x1, self.w0, self.w1))


def test_quantized_wgrad_rounds_inputs_straight_through(hooked):
    """The pre-hook rounds the module input: fprop and wgrad read it, dgrad passes through unchanged."""
    w, x, g = _toy_case()
    _register(*w)
    module = _ToyGroupedLinearModule(*w)
    module.register_forward_pre_hook(expert_nvfp4.round_input_pre_hook(lambda t: t.copy_(_round_input(t))))
    inp = torch.cat([xi.detach() for xi in x]).requires_grad_()
    xq = _round_input(inp.detach())
    out = module(inp * 1, [3, 3])  # a non-leaf, as the dispatcher's output
    out.backward(torch.cat(g))
    assert not torch.equal(xq, inp.detach())
    wq = [_round(wi.detach()) for wi in w]
    torch.testing.assert_close(out, torch.cat([xq[:3] @ wq[0].t(), xq[3:] @ wq[1].t()]), rtol=0, atol=0)
    torch.testing.assert_close(inp.grad, torch.cat([g[0] @ wq[0], g[1] @ wq[1]]), rtol=0, atol=0)
    for i in range(2):
        torch.testing.assert_close(w[i].grad, g[i].t() @ xq[3 * i : 3 * i + 3], rtol=0, atol=0)


def test_in_place_rounding_of_a_saved_input_fails_loudly(hooked):
    """Autograd's version check catches an input that something had already saved for backward."""
    w, x, g = _toy_case()
    module = _ToyGroupedLinearModule(*w)
    module.register_forward_pre_hook(expert_nvfp4.round_input_pre_hook(lambda t: t.copy_(_round_input(t))))
    inp = torch.cat([xi.detach() for xi in x]).requires_grad_()
    pre = inp * 1
    saved = pre * pre  # saves ``pre`` for its backward
    out = module(pre, [3, 3])
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        (out.sum() + saved.sum()).backward()


_SAFE_RECOMPUTE = dict(
    recompute_granularity="full", recompute_modules=None, fine_grained_activation_offloading=False, offload_modules=[]
)


@pytest.mark.parametrize(
    "overrides",
    [
        {"recompute_granularity": "selective", "recompute_modules": ["moe_act"]},
        {"fine_grained_activation_offloading": True, "offload_modules": ["expert_fc1"]},
        {"fine_grained_activation_offloading": True, "offload_modules": ["moe_act"]},
    ],
)
def test_input_rounding_rejects_refilled_or_released_inputs(overrides):
    with pytest.raises(ValueError, match="does not support"):
        expert_nvfp4.check_supported_input_rounding_config(SimpleNamespace(**{**_SAFE_RECOMPUTE, **overrides}))


def test_input_rounding_accepts_full_recompute_and_other_offloads():
    expert_nvfp4.check_supported_input_rounding_config(
        SimpleNamespace(
            **{**_SAFE_RECOMPUTE, "fine_grained_activation_offloading": True, "offload_modules": ["attn_norm"]}
        )
    )


def test_input_rounding_registers_each_module_once(monkeypatch):
    modules = [_ToyGroupedLinearModule(*_toy_case()[0]) for _ in range(2)]
    for m in modules:
        m.config = SimpleNamespace(**_SAFE_RECOMPUTE)
    monkeypatch.setattr(expert_nvfp4, "_INPUT_ROUNDING_MODULES", {})
    monkeypatch.setattr(
        expert_nvfp4, "routed_expert_grouped_linears", lambda *_: [("m0", modules[0]), ("m1", modules[1])]
    )
    assert expert_nvfp4.register_routed_expert_input_rounding([]) == 2
    assert expert_nvfp4.register_routed_expert_input_rounding([]) == 2
    assert [len(m._forward_pre_hooks) for m in modules] == [1, 1]


def _te_style_fprop_gemm(gemm, w, x):
    """Call ``gemm`` the way TE's GroupedLinear fprop does: split views of one input, one output."""
    inp = torch.cat(x)
    out = torch.empty(inp.shape[0], w[0].shape[0])
    gemm(
        list(w),
        list(torch.split(inp, [t.shape[0] for t in x])),
        [out],
        [None, None],
        torch.bfloat16,
        single_output=True,
        m_splits=[t.shape[0] for t in x],
        use_bias=False,
    )
    return inp, out


def test_fprop_replacement_takes_fprop_only(monkeypatch):
    """With an ``fprop``, the routed-expert TN GEMM goes to it with TE's operands; dgrad still rounds."""
    monkeypatch.setattr(expert_nvfp4, "_ROUTED_EXPERT_WEIGHTS", {})
    calls = []

    def fprop(weights, inputs, out, m_splits, bf16_gemm):
        calls.append((weights, inputs, m_splits))
        out.fill_(7.0)

    def gemm(A, B, out, *args, layout="TN", **kwargs):
        out[0].copy_(torch.cat([b @ a for a, b in zip(A, B)]) if layout == "NN" else torch.zeros_like(out[0]))

    wrapped = expert_nvfp4.wrap_grouped_gemm(gemm, _round_all, fprop=fprop)
    w, x, _ = _toy_case()
    _register(*w)
    with torch.no_grad():
        _, out = _te_style_fprop_gemm(wrapped, w, x)
    assert torch.equal(out, torch.full_like(out, 7.0))
    ((weights, inputs, m_splits),) = calls
    assert weights[0] is w[0] and m_splits == [3, 3] and torch.equal(inputs[1], x[1])
    g = [torch.randn(3, 4), torch.randn(3, 4)]
    dgrad = [torch.empty(6, 8)]
    with torch.no_grad():
        wrapped(list(w), g, dgrad, layout="NN")
    torch.testing.assert_close(dgrad[0], torch.cat([g[i] @ _round(w[i].detach()) for i in range(2)]))
    assert len(calls) == 1


def test_fprop_replacement_rejects_other_te_calls(monkeypatch):
    monkeypatch.setattr(expert_nvfp4, "_ROUTED_EXPERT_WEIGHTS", {})
    wrapped = expert_nvfp4.wrap_grouped_gemm(lambda *a, **k: None, _round_all, fprop=lambda *a: None)
    w, x, _ = _toy_case()
    _register(*w)
    out = torch.empty(6, 4)
    with pytest.raises(RuntimeError, match="bias-free"):
        wrapped(
            list(w), list(x), [out], [None, None], torch.bfloat16, single_output=True, m_splits=[3, 3], use_bias=True
        )


def _transformer_config(**overrides):
    base = dict(
        fp8=None, fp4=None, moe_use_grouped_tensor=False, moe_single_grouped_weight=False,
        use_transformer_engine_op_fuser=False,
    )  # fmt: skip
    return SimpleNamespace(**{**base, **overrides})


def test_transformer_config_check_accepts_bf16_per_expert_path():
    expert_nvfp4.check_supported_transformer_config(_transformer_config())


@pytest.mark.parametrize(
    "overrides",
    [
        {"fp8": "e4m3"},
        {"fp4": "e2m1"},
        {"moe_use_grouped_tensor": True},
        {"moe_single_grouped_weight": True},
        {"use_transformer_engine_op_fuser": True},
    ],
)
def test_transformer_config_check_rejects_paths_the_hook_misses(overrides):
    with pytest.raises(ValueError):
        expert_nvfp4.check_supported_transformer_config(_transformer_config(**overrides))


class _Layer(torch.nn.Module):
    """Stand-in for megatron's TransformerLayer: a 1-based global ``layer_number`` and an MoE MLP."""

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number
        self.mlp = _Moe()


class _Moe(torch.nn.Module):
    pass


def test_bf16_layers_are_skipped_by_global_layer_number(monkeypatch):
    """A pipeline stage holding global layers 5..7 under any layout skips exactly the BF16 ones it holds."""
    try:
        import megatron.core.transformer.transformer_layer as transformer_layer

        monkeypatch.setattr(transformer_layer, "TransformerLayer", _Layer)
    except ModuleNotFoundError:  # the skyrl-train extra has no megatron-core: stub the one class it imports
        name = "megatron.core.transformer.transformer_layer"
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            monkeypatch.setitem(sys.modules, ".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
        sys.modules[name].TransformerLayer = _Layer
    chunk = torch.nn.Module()
    chunk.decoder = torch.nn.Module()
    chunk.decoder.layers = torch.nn.ModuleList([_Layer(n) for n in (6, 7, 8)])  # local 0..2 = global 5..7
    found = [name for name, _ in expert_nvfp4.nvfp4_modules([chunk], _Moe, bf16_layers=[0, 7])]
    assert found == ["decoder.layers.0.mlp", "decoder.layers.1.mlp"]
    assert len(list(expert_nvfp4.nvfp4_modules([chunk], _Moe))) == 3


@pytest.mark.parametrize("mode", ["w4a16", "w4a4"])
def test_install_wires_the_mode(monkeypatch, mode):
    """w4a16 hooks the GEMM for the weights only; w4a4 also rounds and packs both expert inputs for the NVFP4 fprop,
    with the same block scale, on any dispatcher."""
    calls = {}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(expert_nvfp4, "install_grouped_gemm_hook", lambda *a, **k: calls.update(hook=(a, k)))
    monkeypatch.setattr(expert_nvfp4, "register_routed_expert_weights", lambda *a: 4)
    monkeypatch.setattr(
        expert_nvfp4, "register_routed_expert_input_rounding", lambda *a, **k: calls.update(rounding=(a, k))
    )
    assert expert_nvfp4.install_expert_nvfp4([], mode, "absmax", [3]) == 4
    assert calls["hook"] == (("absmax",), {"fp4_forward": mode == "w4a4"})
    assert ("rounding" in calls) == (mode == "w4a4")
    if mode == "w4a4":
        assert calls["rounding"] == (([], "absmax"), {"pack_fp4": True, "bf16_layers": [3]})


def test_install_rejects_unknown_modes_and_non_blackwell_devices(monkeypatch):
    with pytest.raises(ValueError, match="expert_nvfp4_mode"):
        expert_nvfp4.install_expert_nvfp4([], "w8a8", "absmax")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (9, 0))
    with pytest.raises(RuntimeError, match="SM100"):
        expert_nvfp4.install_expert_nvfp4([], "w4a16", "absmax")


def test_block_scales_are_checked():
    with pytest.raises(ValueError, match="block_scale"):
        expert_nvfp4.install_grouped_gemm_hook("mse")
    with pytest.raises(ValueError, match="block_scale"):
        expert_nvfp4.register_routed_expert_input_rounding([], "mse")
