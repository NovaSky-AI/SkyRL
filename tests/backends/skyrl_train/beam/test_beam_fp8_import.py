"""Beam import of compressed-tensors FP8_DYNAMIC checkpoints (per-output-channel weight scales)."""

import pytest
import torch

pytest.importorskip("megatron.bridge")

from megatron.bridge.models.beam.beam_bridge import BeamBridge

pytestmark = pytest.mark.megatron

FP8 = torch.float8_e4m3fn


@pytest.fixture
def fp8_state():
    state = {}
    reference = {}
    for name, scale in (
        ("model.layers.0.self_attn.q_proj.weight", 2.0),
        ("model.layers.0.mlp.experts.0.gate_proj.weight", 0.5),
    ):
        # The first row's products expose premature BF16 rounding.
        state[name] = torch.tensor([[1.5, -2.5, 0.5], [-4.0, 0.0, 8.0]], dtype=FP8)
        state[name + "_scale"] = torch.tensor([[1.0078125], [scale]], dtype=torch.bfloat16)
        reference[name] = torch.tensor([[1.51171875, -2.51953125, 0.50390625], [-4 * scale, 0.0, 8 * scale]])
    state["model.layers.0.mlp.gate.weight"] = torch.ones(2, 3, dtype=torch.bfloat16)
    return state, reference


def test_str_param_dequantizes_with_channel_scale(fp8_state):
    state, reference = fp8_state
    name = "model.layers.0.self_attn.q_proj.weight"
    out = BeamBridge().maybe_modify_loaded_hf_weight(name, state)
    assert out.dtype == torch.float32
    torch.testing.assert_close(out, reference[name], rtol=0, atol=0)
    assert out.shape == (2, 3)


def test_dict_param_dequantizes_each_component(fp8_state):
    state, reference = fp8_state
    hf_param = {"gate": "model.layers.0.mlp.experts.0.gate_proj.weight", "q": "model.layers.0.self_attn.q_proj.weight"}
    out = BeamBridge().maybe_modify_loaded_hf_weight(hf_param, state)
    assert set(out) == {"gate", "q"}
    for key, name in hf_param.items():
        torch.testing.assert_close(out[key], reference[name], rtol=0, atol=0)


def test_bf16_tensors_pass_through_untouched(fp8_state):
    state, _ = fp8_state
    name = "model.layers.0.mlp.gate.weight"
    out = BeamBridge().maybe_modify_loaded_hf_weight(name, state)
    assert out is state[name]


def test_fp8_weight_without_scale_is_rejected(fp8_state):
    state, _ = fp8_state
    name = "model.layers.0.self_attn.q_proj.weight"
    del state[name + "_scale"]
    with pytest.raises(ValueError, match="weight_scale"):
        BeamBridge().maybe_modify_loaded_hf_weight(name, state)


def test_fp8_scale_with_wrong_layout_is_rejected(fp8_state):
    state, _ = fp8_state
    name = "model.layers.0.self_attn.q_proj.weight"
    state[name + "_scale"] = torch.ones(1, 3, dtype=torch.bfloat16)  # per-input-channel: not FP8_DYNAMIC
    with pytest.raises(ValueError, match="per-output-channel"):
        BeamBridge().maybe_modify_loaded_hf_weight(name, state)


def test_import_param_names_declare_weight_scale_companions():
    available = {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_proj.weight_scale",
        "model.layers.0.self_attn.k_proj.weight",
    }
    names = BeamBridge.get_hf_import_param_names(
        {"q": "model.layers.0.self_attn.q_proj.weight", "k": "model.layers.0.self_attn.k_proj.weight"}, available
    )
    assert "model.layers.0.self_attn.q_proj.weight_scale" in names
    assert not any(n.endswith("k_proj.weight_scale") for n in names)
    # Without a key set the declaration stays the plain mapping names.
    assert BeamBridge.get_hf_import_param_names("model.layers.0.self_attn.q_proj.weight") == (
        "model.layers.0.self_attn.q_proj.weight",
    )
