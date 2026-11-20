"""SkyRL's vendored ``BeamConfig`` (``patches/transformers/configuration_beam.py``).

Megatron-Bridge's ``BeamBridge`` reads ``rope_parameters`` keyed by layer type (``None`` for the
NoPE full-attention layers); the pinned transformers' RoPE validation cannot take a ``None``
entry. These check that the keyed form survives construction, ``save_pretrained`` and
``AutoConfig`` loading.

Run with:
uv run --isolated --extra skyrl-train --extra dev pytest tests/backends/skyrl_train/patches/transformers/test_configuration_beam.py
"""

import json

import pytest

pytest.importorskip("transformers")

from transformers import AutoConfig  # noqa: E402

from skyrl.backends.skyrl_train.patches.transformers.configuration_beam import (  # noqa: E402
    BeamConfig,
    register_beam_config,
)

KEYED_ROPE = {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}, "full_attention": None}

# reflection/Beam-1.0-501B-A23B's config.json, shortened to 6 layers.
RELEASE_CONFIG = {
    "architectures": ["BeamForCausalLM"],
    "model_type": "beam",
    "vocab_size": 151936,
    "hidden_size": 5120,
    "intermediate_size": 20480,
    "num_hidden_layers": 6,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 1048576,
    "rope_parameters": KEYED_ROPE,
    "sliding_window": 512,
    "moe_intermediate_size": 2560,
    "num_experts_per_tok": 8,
    "num_experts": 256,
    "n_group": 8,
    "topk_group": 4,
    "routed_scaling_factor": 2.5,
    "final_logit_softcapping": 30.0,
    "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
    "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 3,
    "tie_word_embeddings": False,
}


def test_keyed_rope_parameters_survive_auto_config(tmp_path):
    register_beam_config()
    (tmp_path / "config.json").write_text(json.dumps(RELEASE_CONFIG))

    config = AutoConfig.from_pretrained(tmp_path)

    assert isinstance(config, BeamConfig)
    assert config.rope_parameters == KEYED_ROPE
    assert config.layer_types == RELEASE_CONFIG["layer_types"]
    assert config.mlp_layer_types == RELEASE_CONFIG["mlp_layer_types"]
    assert config.final_logit_softcapping == 30.0
    assert config.architectures == ["BeamForCausalLM"]

    config.save_pretrained(tmp_path / "resaved")
    assert AutoConfig.from_pretrained(tmp_path / "resaved").rope_parameters == KEYED_ROPE


def test_defaults_match_transformers_beam_config():
    config = BeamConfig(num_hidden_layers=7)
    # Every sixth layer and the last one run full attention; the first three MLPs are dense.
    assert config.layer_types == ["sliding_attention"] * 5 + ["full_attention"] * 2
    assert config.mlp_layer_types == ["dense"] * 3 + ["sparse"] * 4
    assert config.rope_parameters == KEYED_ROPE
    assert config.architectures == ["BeamForCausalLM"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"layer_types": ["sliding_attention"]}, "layer_types"),
        ({"layer_types": ["global"] * 2}, "layer_types"),
        ({"mlp_layer_types": ["dense", "moe"]}, "mlp_layer_types"),
        ({"num_attention_heads": 6, "num_key_value_heads": 4}, "divisible"),
        (
            {"rope_parameters": {"sliding_attention": {"rope_theta": 1.0}, "full_attention": {"rope_theta": 2.0}}},
            "identical RoPE",
        ),
    ],
)
def test_invalid_configs_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BeamConfig(num_hidden_layers=2, **kwargs)
