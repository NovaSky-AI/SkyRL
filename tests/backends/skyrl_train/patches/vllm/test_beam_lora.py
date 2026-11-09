"""CPU tests for ``patch_beam_lora``: the LoRA-capable Beam class, its registration, and how vLLM's LoRA
loaders name a Megatron-exported Beam adapter.

The model is never built: vLLM objects are created without ``__init__``.
"""

import subprocess
import sys

import pytest
import torch.nn as nn

PATCH = "skyrl.backends.skyrl_train.patches.vllm.patch_beam_lora"


def test_importing_the_patch_does_not_import_vllm():
    # The module is imported from the pyproject entry point in every vLLM process and resolved lazily by the
    # registry, so it must stay importable (and cheap) without building vLLM's model module.
    code = f"import sys, importlib; importlib.import_module({PATCH!r}); print('vllm' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "False"


@pytest.mark.vllm
class TestBeamLoRA:
    @pytest.fixture(autouse=True)
    def vllm_beam(self):
        pytest.importorskip("vllm.models.beam")
        from vllm.models.beam.nvidia.attention import BeamAttention
        from vllm.models.beam.nvidia.model import BeamForCausalLM

        from skyrl.backends.skyrl_train.patches.vllm import patch_beam_lora

        self.patch = patch_beam_lora
        self.base_cls = BeamForCausalLM
        self.attention_cls = BeamAttention
        self.cls = patch_beam_lora.build_beam_lora_class()

    def test_class_supports_lora(self):
        from vllm.model_executor.models.interfaces import supports_lora

        assert issubclass(self.cls, self.base_cls)
        assert self.cls is self.patch.BeamForCausalLM  # resolved through the module __getattr__
        assert self.cls.__name__ == "BeamForCausalLM"
        assert supports_lora(self.cls)
        assert self.cls.embedding_modules == {}
        assert self.cls.is_3d_moe_weight is False  # per-expert adapter keys
        # vLLM's mapping is kept: the 4-slice q/k/v/output gate pack and the 3-slice fallback.
        assert self.cls.packed_modules_mapping["qkv_gate_proj"] == ["q_proj", "k_proj", "v_proj", "output_gate_proj"]
        assert self.cls.packed_modules_mapping["qkv_proj"] == ["q_proj", "k_proj", "v_proj"]
        assert self.cls.packed_modules_mapping["gate_up_proj"] == ["gate_proj", "up_proj"]

    def test_plugin_registers_the_architecture_to_this_module(self, monkeypatch):
        from vllm import ModelRegistry

        monkeypatch.setitem(ModelRegistry.models, "BeamForCausalLM", ModelRegistry.models.get("BeamForCausalLM"))
        self.patch.vllm_general_plugin()
        entry = ModelRegistry.models["BeamForCausalLM"]
        assert (entry.module_name, entry.class_name) == (PATCH, "BeamForCausalLM")

    def _model(self, lora_enabled, attention_modules):
        model = self.cls.__new__(self.cls)
        nn.Module.__init__(model)
        model.model = nn.ModuleList(attention_modules)
        model._skyrl_lora_enabled = lora_enabled
        return model

    def _attention(self):
        attention = self.attention_cls.__new__(self.attention_cls)
        nn.Module.__init__(attention)
        attention.low_latency_projection = True
        attention.merged_qkv_gate = False  # warmup_projection returns early without a projection
        return attention

    def test_lora_disables_the_low_latency_projection(self):
        attentions = [self._attention(), self._attention()]
        self._model(lora_enabled=True, attention_modules=attentions).process_weights_after_loading()
        # The packed GEMM reads qkv_gate_proj.weight directly, which would skip the LoRA wrapper.
        assert [a.low_latency_projection for a in attentions] == [False, False]

    def test_without_lora_vllm_decides(self, monkeypatch):
        calls = []
        monkeypatch.setattr(self.base_cls, "process_weights_after_loading", lambda self: calls.append(self))
        model = self._model(lora_enabled=False, attention_modules=[self._attention()])
        model.process_weights_after_loading()
        assert calls == [model]

    @pytest.mark.parametrize(
        "hf_name, module",
        [
            # The attention output gate is renamed so it packs into qkv_gate_proj / output_gate_proj...
            ("model.layers.0.self_attn.gate_proj", "model.layers.0.self_attn.output_gate_proj"),
            # ...while the dense MLP's gate_proj keeps its name and packs into gate_up_proj.
            ("model.layers.0.mlp.gate_proj", "model.layers.0.mlp.gate_proj"),
            # The rename mapper drops vLLM's stacked mappings, so q/k/v keep their constituent names.
            ("model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.q_proj"),
            ("model.layers.3.mlp.experts.7.gate_proj", "model.layers.3.mlp.experts.7.gate_proj"),
            ("model.layers.3.mlp.experts.7.down_proj", "model.layers.3.mlp.experts.7.down_proj"),
        ],
    )
    def test_adapter_names_through_the_loaders_mapper(self, hf_name, module):
        from vllm.lora.utils import parse_fine_tuned_lora_name

        # What WorkerLoRAManager._load_adapter and patch_lora_in_memory pass to the LoRA model.
        mapper = self.cls.hf_to_vllm_mapper.get_rename_mapper()
        for ab, is_a in (("A", True), ("B", False)):
            assert parse_fine_tuned_lora_name(f"base_model.model.{hf_name}.lora_{ab}.weight", mapper) == (module, is_a)
