"""Beam LoRA export: real adapter parameters, nonzero deltas, and PEFT loading."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("megatron.bridge")

from megatron.bridge.models.beam.beam_bridge import BeamBridge
from megatron.bridge.models.beam.beam_mappings import (
    merge_qkv_gate_weights,
    split_qkv_gate_weights,
)
from megatron.bridge.models.conversion.auto_bridge import AutoBridge
from megatron.bridge.models.conversion.peft_bridge import (
    MegatronPeftBridge,
    infer_target_modules_from_adapter_weights,
)
from megatron.bridge.peft.lora_layers import LinearAdapter

pytestmark = pytest.mark.megatron
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "gate_proj")
BASE_NAMES = [f"model.layers.0.self_attn.{name}.weight" for name in PROJECTIONS]


@pytest.fixture
def adapter_case():
    torch.manual_seed(71)
    config = SimpleNamespace(
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        hidden_size=12,
        num_moe_experts=None,
        attention_output_gate=True,
    )
    adapter = LinearAdapter(torch.nn.Linear(12, 24, bias=False), dim=3, alpha=6)
    # No training: deterministic nonzero B makes a wrong split observable.
    with torch.no_grad():
        adapter.linear_out.weight.normal_(std=0.05)
    model = torch.nn.Module()
    model.config = config
    model.adapter = adapter
    weight = SimpleNamespace(
        global_base_prefix="decoder.layers.0.self_attention.linear_qkv",
        adapter_key=None,
        alpha=6,
        dim=3,
        linear_in_weight=SimpleNamespace(
            weight=adapter.linear_in.weight,
            param_name="decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
        ),
        linear_out_weight=SimpleNamespace(
            weight=adapter.linear_out.weight,
            param_name="decoder.layers.0.self_attention.linear_qkv.adapter.linear_out.weight",
        ),
    )
    bridge = BeamBridge()
    bridge.hf_config = SimpleNamespace(num_hidden_layers=1, mlp_layer_types=["dense"])
    return bridge, model, weight


def _export(bridge, model, weight):
    # Only distributed discovery/gather is stubbed. Streaming, Beam's split,
    # PEFT names/config, and safetensors serialization are production code.
    task = SimpleNamespace(global_base_prefix=weight.global_base_prefix, adapter_key=None)
    with (
        patch.object(bridge, "build_adapter_conversion_tasks", return_value={weight.global_base_prefix: [task]}),
        patch.object(bridge, "materialize_adapter_weights", return_value=[weight]),
    ):
        return list(bridge.stream_adapter_weights_megatron_to_hf([model], show_progress=False))


def test_stream_all_four_projections_and_reconstruct_nonzero_delta(adapter_case):
    bridge, model, weight = adapter_case
    exported = {item.param_name: item.weight for item in _export(bridge, model, weight)}
    assert len(exported) == 8
    pieces = []
    b_pieces = []
    for name, rows in zip(PROJECTIONS, (8, 4, 4, 8)):
        prefix = f"model.layers.0.self_attn.{name}"
        a, b = exported[prefix + ".lora_A.weight"], exported[prefix + ".lora_B.weight"]
        assert a.shape == (3, 12)
        assert b.shape == (rows, 3)
        torch.testing.assert_close(a, model.adapter.linear_in.weight, rtol=0, atol=0)
        b_pieces.append(b)
        pieces.append(2 * (b @ a))
    # Pure packing must be exact. Separate GEMMs may round differently when M changes.
    torch.testing.assert_close(
        merge_qkv_gate_weights(model.config, *b_pieces), model.adapter.linear_out.weight, rtol=0, atol=0
    )
    actual = merge_qkv_gate_weights(model.config, *pieces)
    expected = 2 * (model.adapter.linear_out.weight @ model.adapter.linear_in.weight)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
    assert torch.count_nonzero(actual) > 0


def test_merged_export_adds_each_projection_delta(adapter_case):
    bridge, model, weight = adapter_case
    shapes = (8, 4, 4, 8)
    base = {name: torch.randn(rows, 12) for name, rows in zip(BASE_NAMES, shapes)}
    before = {name: value.clone() for name, value in base.items()}
    merged = bridge._merge_lora_adapter_weights([model], base, [weight])
    delta = 2 * (model.adapter.linear_out.weight @ model.adapter.linear_in.weight)
    for name, piece in zip(BASE_NAMES, split_qkv_gate_weights(model.config, delta)):
        torch.testing.assert_close(merged[name], before[name] + piece, rtol=1e-6, atol=1e-7)


def test_full_size_beam_adapter_geometry_without_allocating_base_model():
    config = SimpleNamespace(num_attention_heads=48, num_query_groups=8, kv_channels=128, hidden_size=5120)
    fused_b = torch.arange(14336 * 16, dtype=torch.float32).reshape(14336, 16)
    pieces = BeamBridge()._split_qkv_linear_out_weight([SimpleNamespace(config=config)], fused_b)
    assert [pieces[name].shape for name in PROJECTIONS] == [(6144, 16), (1024, 16), (1024, 16), (6144, 16)]
    torch.testing.assert_close(
        merge_qkv_gate_weights(config, *(pieces[name] for name in PROJECTIONS)), fused_b, rtol=0, atol=0
    )


def test_other_models_keep_three_way_qkv_and_short_targets():
    names = BASE_NAMES[:3]
    assert MegatronPeftBridge()._is_fused_qkv(names)
    assert not MegatronPeftBridge()._is_fused_qkv(BASE_NAMES)
    assert infer_target_modules_from_adapter_weights(name[:-7] + ".lora_A.weight" for name in names) == [
        "k_proj",
        "q_proj",
        "v_proj",
    ]


def test_gate_targets_do_not_select_unrelated_modules():
    names = ["base_model.model." + name[:-7] + ".lora_A.weight" for name in BASE_NAMES]
    names.append("base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight")
    targets = BeamBridge()._adapter_target_modules(names)
    assert "gate_proj" not in targets
    assert "model.layers.0.self_attn.gate_proj" in targets
    assert "model.layers.0.mlp.gate_proj" in targets
    assert not any("model.layers.1.mlp.gate_proj".endswith(target) for target in targets)


def test_mlp_only_adapter_does_not_select_attention_gate():
    targets = BeamBridge()._adapter_target_modules(["model.layers.0.mlp.gate_proj.lora_A.weight"])
    assert targets == ["model.layers.0.mlp.gate_proj"]
    assert not any("model.layers.0.self_attn.gate_proj".endswith(target) for target in targets)


class ToyHFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = {"tie_word_embeddings": False}
        self.model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        for name, rows in zip(PROJECTIONS, (8, 4, 4, 8)):
            layer.self_attn.add_module(name, torch.nn.Linear(12, rows, bias=False))
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_proj = torch.nn.Linear(12, 20, bias=False)
        self.model.layers = torch.nn.ModuleList([layer])

    def prepare_inputs_for_generation(self, *args, **kwargs):
        return {}

    def forward(self, x=None, **kwargs):
        return x


def test_bridge_adapter_saves_and_loads_with_peft(adapter_case, tmp_path):
    peft = pytest.importorskip("peft")
    from safetensors.torch import load_file

    bridge, model, weight = adapter_case
    facade = SimpleNamespace(
        _model_bridge=bridge, export_adapter_weights=lambda *args, **kwargs: _export(bridge, model, weight)
    )
    with patch("torch.distributed.is_initialized", return_value=False):
        AutoBridge.save_hf_adapter(
            facade,
            [model],
            tmp_path,
            SimpleNamespace(dim=3, alpha=6, dropout=0.0),
            base_model_name_or_path="beam-toy",
            show_progress=False,
        )
    config = json.loads((tmp_path / "adapter_config.json").read_text())
    assert len(config["target_modules"]) == 4
    state = load_file(tmp_path / "adapter_model.safetensors")
    assert len(state) == 8
    loaded = peft.PeftModel.from_pretrained(ToyHFModel(), str(tmp_path))
    layer = loaded.get_base_model().model.layers[0]
    assert not hasattr(layer.mlp.gate_proj, "lora_A")
    x = torch.randn(2, 12)
    pieces = []
    for name in PROJECTIONS:
        projection = getattr(layer.self_attn, name)
        torch.testing.assert_close(projection.lora_A["default"].weight, model.adapter.linear_in.weight, rtol=0, atol=0)
        b = projection.lora_B["default"].weight
        torch.testing.assert_close(
            b, state[f"base_model.model.model.layers.0.self_attn.{name}.lora_B.weight"], rtol=0, atol=0
        )
        pieces.append(projection.lora_B["default"](projection.lora_A["default"](x)) * projection.scaling["default"])
    actual = merge_qkv_gate_weights(model.config, *(piece.T for piece in pieces)).T
    torch.testing.assert_close(actual, model.adapter(x))
