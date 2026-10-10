"""CPU checks for the GLM-5.3-Flash vision-language bridge (``Glm5NextVLBridge``).

Needs megatron-bridge importable (skipped otherwise); builds nothing on GPU.
"""

import pytest
import torch

pytest.importorskip("megatron.bridge")
pytest.importorskip("transformers.models.glm5_next")

from transformers.models.glm5_next.configuration_glm5_next import (  # noqa: E402
    Glm5NextVisionConfig,
)
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    Glm5NextVisionModel,
)

from skyrl.backends.skyrl_train.patches.megatron.glm5_next.bridge import (  # noqa: E402
    GLM5_NEXT_VL_SENTINEL,
    Glm5NextBridge,
    Glm5NextVLBridge,
)


def _tiny_vision_config() -> Glm5NextVisionConfig:
    return Glm5NextVisionConfig(
        depth=2,
        hidden_size=64,
        num_heads=4,
        intermediate_size=128,
        out_hidden_size=96,
        projection_intermediate_size=192,
    )


def _megatron_names(registry) -> list[str]:
    return [m.megatron_param for m in registry.mappings]


def test_vl_bridge_prefixes_every_language_model_mapping():
    text = _megatron_names(Glm5NextBridge().mapping_registry())
    vl = _megatron_names(Glm5NextVLBridge().mapping_registry())
    # The registry appends derived layernorm aliases at the end, so compare as sets.
    assert {n for n in vl if not n.startswith("visual.")} == {f"language_model.{n}" for n in text}
    assert len(vl) == len(text) + 1
    assert [n for n in vl if n.startswith("visual.")] == ["visual.**"]


def test_vl_bridge_maps_every_vision_parameter_one_to_one():
    with torch.device("meta"):
        visual = Glm5NextVisionModel._from_config(_tiny_vision_config(), attn_implementation="sdpa")
    registry = Glm5NextVLBridge().mapping_registry()
    for name, _ in visual.named_parameters():
        mapping = registry.megatron_to_hf_lookup(f"visual.{name}")
        assert mapping is not None, name
        assert mapping.hf_param == f"model.visual.{name}"


def test_text_bridge_is_unchanged_by_default():
    # The text-only path (language_model_only=true) must keep unprefixed GPTModel names.
    names = _megatron_names(Glm5NextBridge().mapping_registry())
    assert "embedding.word_embeddings.weight" in names
    assert not any(n.startswith(("language_model.", "visual.")) for n in names)


def test_force_vl_bridge_only_for_glm5_next():
    from types import SimpleNamespace

    from skyrl.backends.skyrl_train.workers.megatron.model_bridges import (
        maybe_force_glm5_next_vl_bridge,
    )

    def bridge(arch):
        return SimpleNamespace(hf_pretrained=SimpleNamespace(config=SimpleNamespace(architectures=[arch])))

    glm = bridge("Glm5NextForConditionalGeneration")
    assert maybe_force_glm5_next_vl_bridge(glm, SimpleNamespace(model_type="glm5_next"))
    assert glm.hf_pretrained.config.architectures == [GLM5_NEXT_VL_SENTINEL]

    qwen = bridge("Qwen3VLForConditionalGeneration")
    assert not maybe_force_glm5_next_vl_bridge(qwen, SimpleNamespace(model_type="qwen3_vl"))
    assert qwen.hf_pretrained.config.architectures == ["Qwen3VLForConditionalGeneration"]
