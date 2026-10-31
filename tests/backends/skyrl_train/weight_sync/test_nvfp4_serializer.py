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
        {"NVTE_NVFP4_4OVER6_E4M3_USE_256": "all"},  # vLLM's per-token activations use the 448 bound
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
