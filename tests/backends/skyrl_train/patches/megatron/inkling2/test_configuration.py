"""Inkling2 config classes read the release ``config.json`` and the pre-release one."""

import json

import pytest

pytestmark = pytest.mark.megatron

transformers = pytest.importorskip("transformers")

from skyrl.backends.skyrl_train.patches.megatron.inkling2 import (  # noqa: E402
    configuration,
)

LEGACY = {
    "model_type": "inkling2_mm_model",
    "architectures": ["Inkling2ForConditionalGeneration"],
    "text_config": {
        "num_hidden_layers": 8,
        "n_routed_experts": 64,
        "n_layers_per_attnres_block": 6,
        "log_scaling_n_floor": 8192,
    },
}


def _load(tmp_path, config: dict):
    configuration.register()
    (tmp_path / "config.json").write_text(json.dumps(config))
    return transformers.AutoConfig.from_pretrained(tmp_path)


def test_release_config_names(tmp_path):
    config = _load(tmp_path, configuration.release_config(LEGACY))
    assert config.model_type == "inkling2"
    text = config.get_text_config()
    assert text.n_layers_per_attn_res_block == 6
    assert text.log_scaling_n_floor == 8192  # values are kept
    assert text.num_hidden_layers == 8 and text.n_routed_experts == 64


def test_legacy_config_still_loads(tmp_path):
    config = _load(tmp_path, LEGACY)
    assert config.model_type == "inkling2_mm_model"
    assert config.get_text_config().n_layers_per_attn_res_block == 6


def test_release_config_conversion():
    converted = configuration.release_config(LEGACY)
    text = converted["text_config"]
    assert converted["model_type"] == "inkling2"
    assert text["model_type"] == "inkling2_model"
    assert "n_layers_per_attnres_block" not in text and text["n_layers_per_attn_res_block"] == 6
    assert LEGACY["text_config"]["n_layers_per_attnres_block"] == 6  # input untouched
    assert configuration.release_config(converted) == converted
