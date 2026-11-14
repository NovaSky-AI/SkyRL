"""CPU tests for the serialized NVFP4 expert weight sync (stream transform, quantizer call, vLLM overrides)."""

import pytest
import torch

from skyrl.backends.skyrl_train.weight_sync import nvfp4

_SETTINGS = nvfp4.Nvfp4WeightQuantSettings("four_over_six")


@pytest.fixture
def fake_quantizer(monkeypatch):
    """Replace the FlashInfer kernel (GPU-only) with a shape-correct stub that records the quantized matrices."""
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
    serializer = nvfp4.Nvfp4ExpertSerializer(_SETTINGS)
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
    serializer = nvfp4.Nvfp4ExpertSerializer(_SETTINGS, bf16_layers=[3])
    stream = [
        ("model.embed_tokens.weight", torch.zeros(8, 32)),
        _expert(0, 0, "down_proj", rows=32, cols=16),
        _expert(3, 0, "gate_proj"),  # last layer: BF16 layer, passed through untouched
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
    serializer = nvfp4.Nvfp4ExpertSerializer(_SETTINGS)
    list(serializer(*_expert(0, 0, "gate_proj")))
    with pytest.raises(RuntimeError, match="unpaired"):
        serializer.finish()


def test_vllm_engine_overrides_ignore_bf16_layers():
    overrides = nvfp4.get_nvfp4_vllm_engine_overrides(bf16_layers=range(41, 48))
    assert overrides["quantization"] == "modelopt_fp4" and overrides["load_format"] == "dummy"
    assert overrides["quantization_config"] == {"moe": {"activation": "nvfp4_per_token"}}
    qcfg = overrides["hf_overrides"]["quantization_config"]
    assert qcfg["quant_algo"] == "NVFP4" and qcfg["group_size"] == 16
    tail = [p for p in qcfg["ignore"] if ".mlp.experts" in p]
    assert tail == [f"*layers.{i}.mlp.experts*" for i in range(41, 48)]


def test_bf16_layers_follow_the_fp8_first_last_layer_knobs():
    """``num_layers_at_{start,end}_in_bf16`` (shared with FP8 training) pick the BF16 expert layers on both sides."""
    from skyrl.train.config.config import MegatronConfig

    assert nvfp4.nvfp4_bf16_layers(48, MegatronConfig(expert_nvfp4=True)) == []
    cfg = MegatronConfig(expert_nvfp4=True, num_layers_at_end_in_bf16=7)
    assert nvfp4.nvfp4_bf16_layers(48, cfg) == list(range(41, 48))
    cfg = MegatronConfig(expert_nvfp4=True, num_layers_at_start_in_bf16=2, num_layers_at_end_in_bf16=1)
    assert nvfp4.nvfp4_bf16_layers(48, cfg) == [0, 1, 47]
    # An explicit transformer_config_kwargs entry wins, as it does for Megatron itself.
    cfg = MegatronConfig(expert_nvfp4=True, transformer_config_kwargs={"first_last_layers_bf16": True})
    assert nvfp4.nvfp4_bf16_layers(4, cfg) == []
    cfg = MegatronConfig(
        expert_nvfp4=True,
        transformer_config_kwargs={"first_last_layers_bf16": True, "num_layers_at_end_in_bf16": 2},
    )
    assert nvfp4.nvfp4_bf16_layers(4, cfg) == [2, 3]

    class _Provider:  # Megatron's TransformerConfig fields
        first_last_layers_bf16, num_layers_at_start_in_bf16, num_layers_at_end_in_bf16 = True, 1, 1

    assert nvfp4.nvfp4_bf16_layers(4, _Provider()) == [0, 3]
    _Provider.first_last_layers_bf16 = False  # Megatron's 1/1 defaults apply only when enabled
    assert nvfp4.nvfp4_bf16_layers(4, _Provider()) == []
    with pytest.raises(ValueError, match="at least one"):
        nvfp4.nvfp4_bf16_layers(4, MegatronConfig(expert_nvfp4=True, num_layers_at_end_in_bf16=4))


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
        _expert(3, 0, "gate_proj"),  # BF16 layer
        _expert(3, 0, "up_proj"),
        ("model.norm.weight", torch.ones(32)),
    ]


def test_spec_serializer_declares_the_real_stream(fake_quantizer):
    """The data-free pass (what ``metadata()`` runs on every rank) enumerates exactly what quantizing yields."""
    real = _run(nvfp4.Nvfp4ExpertSerializer(_SETTINGS, bf16_layers=[3]), _mixed_stream())
    n_quantized = len(fake_quantizer)
    specs = [(name, nvfp4.TensorSpec(t.dtype, tuple(t.shape))) for name, t in _mixed_stream()]
    declared = _run(nvfp4.Nvfp4ExpertSpecSerializer(_SETTINGS, bf16_layers=[3]), specs)
    assert len(fake_quantizer) == n_quantized  # the spec pass never touched the quantizer
    assert [(n, s.dtype, tuple(s.shape)) for n, s in declared] == [(n, t.dtype, tuple(t.shape)) for n, t in real]


def test_spec_serializer_rejects_what_the_kernel_rejects():
    serializer = nvfp4.Nvfp4ExpertSpecSerializer(_SETTINGS)
    with pytest.raises(ValueError, match="divisible"):
        list(serializer("model.layers.0.mlp.experts.0.down_proj.weight", nvfp4.TensorSpec(torch.bfloat16, (4, 24))))
    list(serializer("model.layers.0.mlp.experts.1.gate_proj.weight", nvfp4.TensorSpec(torch.bfloat16, (4, 32))))
    with pytest.raises(ValueError, match="must match"):
        list(serializer("model.layers.0.mlp.experts.1.up_proj.weight", nvfp4.TensorSpec(torch.bfloat16, (8, 32))))


def _fake_quantize_experts(calls=None):
    def quantize_experts(w, gs, block_scale):
        if calls is not None:
            calls.append((w.clone(), gs.clone(), block_scale))
        e, n, k = w.shape
        return torch.zeros(e, n, k // 2, dtype=torch.uint8), torch.zeros(e, n, k // 16, dtype=torch.float8_e4m3fn)

    return quantize_experts


def test_spec_matches_real_quantize_outputs(monkeypatch):
    """Pins ``_emit_spec`` to what ``quantize_nvfp4_matrix`` + ``_emit`` produce."""
    monkeypatch.setattr(nvfp4, "quantize_experts", _fake_quantize_experts())
    for rows in (16, 20, 3):
        w = torch.randn(rows, 48)
        real = list(nvfp4._emit("p", "down_proj", *nvfp4.quantize_nvfp4_matrix(w, _SETTINGS)))
        spec = list(nvfp4._emit_spec("p", "down_proj", rows, 48))
        assert [(n, t.dtype, tuple(t.shape)) for n, t in real] == [(n, s.dtype, tuple(s.shape)) for n, s in spec]


@pytest.mark.parametrize("block_scale", ["absmax", "four_over_six"])
def test_quantizes_with_vllms_global_scale_over_the_whole_matrix(monkeypatch, block_scale):
    """One ``quantize_experts`` call per matrix with vLLM's global scale over the whole ``[gate; up]`` matrix and the
    configured block scale, so the bytes are the trainer's round trip's."""
    calls = []
    monkeypatch.setattr(nvfp4, "quantize_experts", _fake_quantize_experts(calls))
    w = torch.cat((torch.full((6, 32), 1.0), torch.full((6, 32), -3.0)))
    packed, scales, decode = nvfp4.quantize_nvfp4_matrix(w, nvfp4.Nvfp4WeightQuantSettings(block_scale))
    ((qw, gs, bs),) = calls
    assert torch.equal(qw, w[None]) and bs == block_scale
    assert gs.item() == pytest.approx(6 * 448 / 3.0) and decode.item() == pytest.approx(3.0 / (6 * 448))
    assert packed.shape == (12, 16) and scales.shape == (12, 2)


def test_settings_reject_unknown_block_scales():
    with pytest.raises(ValueError, match="block_scale"):
        nvfp4.Nvfp4WeightQuantSettings("mse")
