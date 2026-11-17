"""The Qwen3.5 LM-only bridges (``language_model_only=True``) map the checkpoint's top-level MTP head."""

import pytest

pytest.importorskip("megatron.bridge")

from skyrl.backends.skyrl_train.workers.megatron.model_bridges import (  # noqa: E402
    Qwen35DenseLMBridge,
    Qwen35MoELMBridge,
)

pytestmark = pytest.mark.megatron

MTP_HEAD = {
    "mtp.layers.0.eh_proj.weight": "mtp.fc.weight",
    "mtp.layers.0.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
    "mtp.layers.0.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.final_layernorm.weight": "mtp.norm.weight",
    "mtp.layers.0.mtp_model_layer.self_attention.linear_proj.weight": "mtp.layers.0.self_attn.o_proj.weight",
}


@pytest.mark.parametrize("bridge_cls", [Qwen35DenseLMBridge, Qwen35MoELMBridge])
def test_lm_only_bridge_maps_the_top_level_mtp_head(bridge_cls):
    registry = bridge_cls().mapping_registry()
    for megatron_name, hf_name in MTP_HEAD.items():
        mapping = registry.megatron_to_hf_lookup(megatron_name)
        assert mapping is not None, megatron_name
        assert mapping.hf_param == hf_name
    qkv = registry.megatron_to_hf_lookup("mtp.layers.0.mtp_model_layer.self_attention.linear_qkv.weight")
    assert qkv.hf_param["q"] == "mtp.layers.0.self_attn.q_proj.weight"


@pytest.mark.parametrize("bridge_cls", [Qwen35DenseLMBridge, Qwen35MoELMBridge])
def test_lm_only_bridge_keeps_the_language_model_prefix(bridge_cls):
    registry = bridge_cls().mapping_registry()
    assert registry.megatron_to_hf_lookup("embedding.word_embeddings.weight").hf_param == (
        "model.language_model.embed_tokens.weight"
    )
