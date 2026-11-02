"""CPU tests for the serialized NVFP4 expert weight sync (stream transform, env contract, vLLM overrides)."""

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync import nvfp4

_SETTINGS = nvfp4.Nvfp4WeightQuantSettings(
    use_4over6=True, e4m3_max=448, err_mode="MSE", row_scaled_activation=True, backward_override="dequantized"
)


@pytest.fixture
def fake_quantizer(monkeypatch):
    """Replace the TE kernel (GPU-only) with a shape-correct stub that records the quantized matrices."""
    calls = []

    def quantize(weight, settings):
        calls.append(weight.clone())
        rows, cols = weight.shape
        packed = torch.full((rows, cols // 2), 7, dtype=torch.uint8)
        block_scale = torch.zeros(rows, cols // 16, dtype=torch.uint8).view(torch.float8_e4m3fn)
        return packed, block_scale, torch.tensor(float(len(calls)))

    monkeypatch.setattr(nvfp4, "quantize_nvfp4_matrix", quantize)
    return calls


def _expert(layer, expert, proj, rows=6, cols=32, fill=0.0):
    return f"model.layers.{layer}.mlp.experts.{expert}.{proj}.weight", torch.full((rows, cols), fill)


def _run(serializer, stream):
    out = [(name, tensor) for name0, tensor0 in stream for name, tensor in serializer(name0, tensor0)]
    serializer.finish()
    return out


def test_gate_up_quantized_as_one_matrix_in_any_order(fake_quantizer):
    serializer = nvfp4.Nvfp4ExpertSerializer(_SETTINGS, num_layers=4, high_precision_last_layers=0)
    out = dict(
        _run(
            serializer,
            [
                _expert(0, 0, "up_proj", fill=2.0),
                _expert(0, 1, "gate_proj"),
                _expert(0, 0, "gate_proj", fill=1.0),
                _expert(0, 1, "up_proj"),
            ],
        )
    )
    # Expert 0's up arrived first; the pair is quantized once as [gate; up].
    assert torch.equal(fake_quantizer[0][:6], torch.ones(6, 32)) and torch.equal(
        fake_quantizer[0][6:], torch.full((6, 32), 2.0)
    )
    prefix = "model.layers.0.mlp.experts.0"
    assert out[f"{prefix}.gate_proj.weight"].dtype == torch.uint8
    assert tuple(out[f"{prefix}.gate_proj.weight"].shape) == (6, 16)
    assert out[f"{prefix}.gate_proj.weight_scale"].dtype == torch.float8_e4m3fn
    assert tuple(out[f"{prefix}.up_proj.weight_scale"].shape) == (6, 2)
    # gate and up share one global scale (vLLM fuses w13).
    assert torch.equal(out[f"{prefix}.gate_proj.weight_scale_2"], out[f"{prefix}.up_proj.weight_scale_2"])
    assert out[f"{prefix}.gate_proj.weight_scale_2"].shape == ()
    assert out[f"{prefix}.up_proj.input_scale"].item() == 1.0


def test_down_proj_and_passthrough(fake_quantizer):
    serializer = nvfp4.Nvfp4ExpertSerializer(_SETTINGS, num_layers=4, high_precision_last_layers=1)
    stream = [
        ("model.embed_tokens.weight", torch.zeros(8, 32)),
        _expert(0, 0, "down_proj", rows=32, cols=16),
        _expert(3, 0, "gate_proj"),  # last layer: BF16 tail, passed through untouched
        ("model.layers.0.mlp.gate.weight", torch.zeros(4, 32)),  # router stays BF16
    ]
    out = _run(serializer, stream)
    names = [name for name, _ in out]
    assert names == [
        "model.embed_tokens.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight_scale",
        "model.layers.0.mlp.experts.0.down_proj.weight_scale_2",
        "model.layers.0.mlp.experts.0.down_proj.input_scale",
        "model.layers.3.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.gate.weight",
    ]
    assert out[5][1].dtype == torch.float32 and len(fake_quantizer) == 1


def test_unpaired_gate_is_an_error(fake_quantizer):
    serializer = nvfp4.Nvfp4ExpertSerializer(_SETTINGS, num_layers=4, high_precision_last_layers=0)
    list(serializer(*_expert(0, 0, "gate_proj")))
    with pytest.raises(RuntimeError, match="unpaired"):
        serializer.finish()


def test_runtime_env_derives_matching_flashinfer_settings():
    env = nvfp4.nvfp4_runtime_env({})
    assert env["NVTE_BACKWARD_OVERRIDE"] == "dequantized"
    assert env["NVTE_NVFP4_ROW_SCALED_ACTIVATION"] == "1"
    assert env["NVTE_NVFP4_4OVER6"] == "all" and env["FLASHINFER_NVFP4_4OVER6"] == "1"
    assert env["FLASHINFER_NVFP4_4OVER6_ERR_MODE"] == env["NVTE_NVFP4_4OVER6_ERR_MODE"]
    assert env["FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH"] == "1"

    no_4over6 = nvfp4.nvfp4_runtime_env({"NVTE_NVFP4_4OVER6": "none"})
    assert no_4over6["FLASHINFER_NVFP4_4OVER6"] == "0"


def test_runtime_env_pure_recipe_leaves_te_defaults():
    env = nvfp4.nvfp4_runtime_env({"SKYRL_NVFP4_RECIPE": "pure"})
    assert "NVTE_BACKWARD_OVERRIDE" not in env and "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING" not in env
    assert env["NVTE_NVFP4_DISABLE_RHT"] == "1"  # TE rejects RHT with row-scaled activations
    assert env["NVTE_NVFP4_4OVER6"] == "none" and env["FLASHINFER_NVFP4_4OVER6"] == "0"


@pytest.mark.parametrize(
    "environ",
    [
        {"FLASHINFER_NVFP4_4OVER6": "0"},  # contradicts the TE 4/6 setting
        {"NVTE_NVFP4_4OVER6_E4M3_USE_256": "bogus"},
        # 256 on activations derives FLASHINFER_NVFP4_4OVER6_E4M3_USE_256=1; an explicit 0 contradicts it.
        {"NVTE_NVFP4_4OVER6_E4M3_USE_256": "all", "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "0"},
        {"NVTE_NVFP4_4OVER6_E4M3_USE_256": "none", "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": "1"},
        {"NVTE_NVFP4_ROW_SCALED_ACTIVATION": "0"},
        {"SKYRL_NVFP4_RECIPE": "bogus"},
    ],
)
def test_runtime_env_rejects_sampler_mismatch(environ):
    with pytest.raises(ValueError):
        nvfp4.nvfp4_runtime_env(environ)


def test_vllm_engine_overrides_ignore_bf16_tail():
    overrides = nvfp4.get_nvfp4_vllm_engine_overrides(num_layers=48, high_precision_last_layers=7)
    assert overrides["quantization"] == "modelopt_fp4" and overrides["load_format"] == "dummy"
    assert overrides["quantization_config"] == {"moe": {"activation": "nvfp4_per_token"}}
    qcfg = overrides["hf_overrides"]["quantization_config"]
    assert qcfg["quant_algo"] == "NVFP4" and qcfg["group_size"] == 16
    tail = [p for p in qcfg["ignore"] if ".mlp.experts" in p]
    assert tail == [f"*layers.{i}.mlp.experts*" for i in range(41, 48)]


def _mixed_stream():
    return [
        ("model.embed_tokens.weight", torch.zeros(8, 32, dtype=torch.bfloat16)),
        _expert(0, 0, "up_proj"),
        _expert(0, 1, "gate_proj"),
        _expert(0, 0, "down_proj", rows=32, cols=16),
        _expert(0, 0, "gate_proj"),
        _expert(0, 1, "up_proj"),
        _expert(0, 1, "down_proj", rows=32, cols=16),
        ("model.layers.0.mlp.gate.weight", torch.zeros(4, 32)),
        _expert(3, 0, "gate_proj"),  # BF16 tail
        _expert(3, 0, "up_proj"),
        ("model.norm.weight", torch.ones(32)),
    ]


def test_spec_serializer_declares_the_real_stream(fake_quantizer):
    """The data-free pass (what ``metadata()`` runs on every rank) enumerates exactly what quantizing yields."""
    real = _run(nvfp4.Nvfp4ExpertSerializer(_SETTINGS, num_layers=4, high_precision_last_layers=1), _mixed_stream())
    n_quantized = len(fake_quantizer)
    specs = [(name, nvfp4.TensorSpec(t.dtype, tuple(t.shape))) for name, t in _mixed_stream()]
    declared = _run(nvfp4.Nvfp4ExpertSpecSerializer(_SETTINGS, num_layers=4, high_precision_last_layers=1), specs)
    assert len(fake_quantizer) == n_quantized  # the spec pass never touched the quantizer
    assert [(n, s.dtype, tuple(s.shape)) for n, s in declared] == [(n, t.dtype, tuple(t.shape)) for n, t in real]


def test_spec_serializer_rejects_what_the_kernel_rejects():
    serializer = nvfp4.Nvfp4ExpertSpecSerializer(_SETTINGS, num_layers=4, high_precision_last_layers=0)
    with pytest.raises(ValueError, match="divisible"):
        list(serializer("model.layers.0.mlp.experts.0.down_proj.weight", nvfp4.TensorSpec(torch.bfloat16, (4, 24))))
    list(serializer("model.layers.0.mlp.experts.1.gate_proj.weight", nvfp4.TensorSpec(torch.bfloat16, (4, 32))))
    with pytest.raises(ValueError, match="must match"):
        list(serializer("model.layers.0.mlp.experts.1.up_proj.weight", nvfp4.TensorSpec(torch.bfloat16, (8, 32))))


class _FakeNVFP4Quantizer:
    """Deterministic CPU stand-in for TE's NVFP4Quantizer: quantize() is a pure function of its input."""

    constructed = 0

    def __init__(self, **kwargs):
        type(self).constructed += 1
        self.kwargs = kwargs

    def quantize(self, src):
        from types import SimpleNamespace

        rows, cols = src.shape
        packed = (src.float().abs() * 7).to(torch.uint8)[:, : cols // 2]
        scale = (src.float().abs().reshape(rows, cols // 16, 16).amax(-1) * 3).to(torch.uint8)
        return SimpleNamespace(
            _rowwise_data=packed.contiguous(),
            _rowwise_scale_inv=scale.contiguous(),
            _amax_rowwise=src.float().abs().amax().reshape(1),
        )


def _bits(t):
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


@pytest.fixture
def fake_te(monkeypatch):
    import sys
    import types

    mod = types.ModuleType("transformer_engine.pytorch.tensor.nvfp4_tensor")
    mod.NVFP4Quantizer = _FakeNVFP4Quantizer
    for name in ("transformer_engine", "transformer_engine.pytorch", "transformer_engine.pytorch.tensor"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.tensor.nvfp4_tensor", mod)
    monkeypatch.setattr(nvfp4, "_QUANTIZER_CACHE", {})
    _FakeNVFP4Quantizer.constructed = 0
    return _FakeNVFP4Quantizer


def test_quantizer_is_built_once_and_reuse_is_identical(fake_te):
    gen = torch.Generator().manual_seed(0)
    weights = [torch.randn(20, 64, generator=gen) for _ in range(5)]
    cached = [nvfp4.quantize_nvfp4_matrix(w, _SETTINGS) for w in weights]
    assert fake_te.constructed == 1
    kwargs = next(iter(nvfp4._QUANTIZER_CACHE.values())).kwargs
    assert kwargs["nvfp4_use_4over6"] is True and kwargs["nvfp4_4over6_err_mode"] == "MSE"
    assert kwargs["with_rht"] is False and kwargs["stochastic_rounding"] is False

    for w, got in zip(weights, cached):
        nvfp4._QUANTIZER_CACHE.clear()  # a fresh quantizer per matrix, as before the cache
        fresh = nvfp4.quantize_nvfp4_matrix(w, _SETTINGS)
        for a, b in zip(got, fresh):
            assert a.dtype == b.dtype and a.shape == b.shape
            assert torch.equal(_bits(a), _bits(b))

    other = nvfp4.Nvfp4WeightQuantSettings(
        use_4over6=False, e4m3_max=448, err_mode="MAE", row_scaled_activation=True, backward_override=None
    )
    nvfp4.quantize_nvfp4_matrix(weights[0], other)
    assert len(nvfp4._QUANTIZER_CACHE) == 2  # different settings never share a quantizer


def test_spec_matches_real_quantize_outputs(fake_te):
    """Pins ``_emit_spec`` to what ``quantize_nvfp4_matrix`` + ``_emit`` produce (incl. the 16-row padding path)."""
    for rows in (16, 20, 3):
        w = torch.randn(rows, 48)
        real = list(nvfp4._emit("p", "down_proj", *nvfp4.quantize_nvfp4_matrix(w, _SETTINGS)))
        spec = list(nvfp4._emit_spec("p", "down_proj", rows, 48))
        assert [(n, t.dtype, tuple(t.shape)) for n, t in real] == [(n, s.dtype, tuple(s.shape)) for n, s in spec]


@pytest.mark.parametrize(
    "four6, use_256, fi_4over6, fi_256",
    [
        ("none", "none", "0", "0"),
        ("none", "weights", "0", "0"),
        ("none", "activations", "0", "0"),  # no activation 4/6: the bound does not reach the sampler
        ("none", "all", "0", "0"),
        ("weights", "none", "0", "0"),
        ("weights", "weights", "0", "0"),
        ("weights", "activations", "0", "0"),
        ("weights", "all", "0", "0"),
        ("activations", "none", "1", "0"),
        ("activations", "weights", "1", "0"),  # weights' 256 travels in the weights' own decode scales
        ("activations", "activations", "1", "1"),
        ("activations", "all", "1", "1"),
        ("all", "none", "1", "0"),
        ("all", "weights", "1", "0"),
        ("all", "activations", "1", "1"),
        ("all", "all", "1", "1"),
    ],
)
def test_runtime_env_e4m3_256_derivation(four6, use_256, fi_4over6, fi_256):
    env = nvfp4.nvfp4_runtime_env({"NVTE_NVFP4_4OVER6": four6, "NVTE_NVFP4_4OVER6_E4M3_USE_256": use_256})
    assert env["NVTE_NVFP4_4OVER6"] == four6 and env["NVTE_NVFP4_4OVER6_E4M3_USE_256"] == use_256
    assert env["FLASHINFER_NVFP4_4OVER6"] == fi_4over6
    assert env["FLASHINFER_NVFP4_4OVER6_E4M3_USE_256"] == fi_256
    # A matching explicit FlashInfer setting is accepted.
    same = nvfp4.nvfp4_runtime_env(
        {
            "NVTE_NVFP4_4OVER6": four6,
            "NVTE_NVFP4_4OVER6_E4M3_USE_256": use_256,
            "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256": fi_256,
        }
    )
    assert same == env


def test_runtime_env_defaults_keep_448():
    env = nvfp4.nvfp4_runtime_env({})
    assert env["NVTE_NVFP4_4OVER6_E4M3_USE_256"] == "none"
    assert env["FLASHINFER_NVFP4_4OVER6_E4M3_USE_256"] == "0"
