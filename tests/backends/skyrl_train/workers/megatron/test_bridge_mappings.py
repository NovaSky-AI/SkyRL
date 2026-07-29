import sys
from types import ModuleType

from skyrl.backends.skyrl_train.quantization.megatron import (
    get_packed_qwen3_moe_conversion_tasks,
)


def test_qwen3_weight_sync_replaces_per_expert_mappings_with_packed_mappings(monkeypatch):
    class AutoMapping:
        def __init__(self, megatron_param, hf_param):
            self.megatron_param = megatron_param
            self.hf_param = hf_param

    class GatedMLPMapping(AutoMapping):
        def __init__(self, megatron_param, gate, up):
            super().__init__(megatron_param, {"gate": gate, "up": up})

    class FusedExpertMapping(AutoMapping):
        pass

    class FusedGatedExpertMapping(AutoMapping):
        pass

    class MegatronMappingRegistry:
        def __init__(self, *mappings):
            self.mappings = list(mappings)

        def __iter__(self):
            return iter(self.mappings)

        def get_all_mappings(self):
            return self.mappings.copy()

    module_names = (
        "megatron",
        "megatron.bridge",
        "megatron.bridge.models",
        "megatron.bridge.models.conversion",
    )
    for module_name in module_names:
        module = ModuleType(module_name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, module_name, module)

    mapping_registry_module = ModuleType("megatron.bridge.models.conversion.mapping_registry")
    mapping_registry_module.MegatronMappingRegistry = MegatronMappingRegistry
    monkeypatch.setitem(sys.modules, mapping_registry_module.__name__, mapping_registry_module)
    param_mapping_module = ModuleType("megatron.bridge.models.conversion.param_mapping")
    param_mapping_module.FusedExpertMapping = FusedExpertMapping
    param_mapping_module.FusedGatedExpertMapping = FusedGatedExpertMapping
    monkeypatch.setitem(sys.modules, param_mapping_module.__name__, param_mapping_module)

    standard_registry = MegatronMappingRegistry(
        AutoMapping("embedding.word_embeddings.weight", "model.embed_tokens.weight"),
        GatedMLPMapping(
            "decoder.layers.*.mlp.experts.linear_fc1.weight*",
            gate="model.layers.*.mlp.experts.*.gate_proj.weight",
            up="model.layers.*.mlp.experts.*.up_proj.weight",
        ),
        AutoMapping(
            "decoder.layers.*.mlp.experts.linear_fc2.weight*",
            "model.layers.*.mlp.experts.*.down_proj.weight",
        ),
    )

    class FakeModelBridge:
        def mapping_registry(self):
            return standard_registry

        def build_conversion_tasks(self, _hf_pretrained, _model):
            return self.mapping_registry().get_all_mappings()

    class FakeAutoBridge:
        _model_bridge = FakeModelBridge()
        hf_pretrained = object()

    mappings = get_packed_qwen3_moe_conversion_tasks(FakeAutoBridge(), object())

    assert any(isinstance(mapping, FusedGatedExpertMapping) for mapping in mappings)
    assert any(isinstance(mapping, FusedExpertMapping) for mapping in mappings)
    assert any(mapping.megatron_param == "embedding.word_embeddings.weight" for mapping in mappings)
    assert not any(
        isinstance(mapping, GatedMLPMapping)
        and not isinstance(mapping, FusedGatedExpertMapping)
        for mapping in mappings
    )
