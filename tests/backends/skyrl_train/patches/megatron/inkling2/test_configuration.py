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


LOOPED = {
    "num_hidden_layers": 12,
    "n_layers_per_attn_res_block": 3,
    "loop_config": {"core_start": 3, "core_end": 9, "num_passes": 3, "gate_rank": 4, "is_varloop": False},
}
LOOP = LOOPED["loop_config"]


def test_loop_config_helpers(tmp_path):
    config = _load(tmp_path, {"model_type": "inkling2_loop", "text_config": LOOPED})
    assert config.model_type == "inkling2_loop"
    text = config.get_text_config()
    assert text.loop_core_layers == range(3, 9)
    assert text.num_layer_visits == 12 + 2 * 6
    order = configuration.layer_run_order(text.loop_config, 12)
    assert len(order) == text.num_layer_visits
    assert order[:4] == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert order[9:11] == [(3, 1), (4, 1)] and order[-4:] == [(8, 2), (9, 0), (10, 0), (11, 0)]
    unlooped = configuration.Inkling2TextConfig(num_hidden_layers=12)
    assert unlooped.loop_config is None and unlooped.loop_core_layers == range(0)
    assert unlooped.num_layer_visits == 12
    assert configuration.layer_run_order(None, 3) == [(0, 0), (1, 0), (2, 0)]


@pytest.mark.parametrize(
    "loop_config, match",
    [
        ({**LOOP, "gate_rank": 4.0}, "integer fields"),
        ({k: v for k, v in LOOP.items() if k != "gate_rank"}, "integer fields"),
        # Configs from before is_varloop.
        ({k: v for k, v in LOOP.items() if k != "is_varloop"}, "re-download"),
        ({**LOOP, "is_varloop": 0}, "boolean"),
        ({**LOOP, "core_end": 15}, "within"),
        ({**LOOP, "core_start": 2}, "align"),
        ({**LOOP, "num_passes": 1}, "two passes"),
        ({**LOOP, "gate_rank": 0}, "gate rank"),
    ],
)
def test_loop_config_validation(loop_config, match):
    with pytest.raises(ValueError, match=match):
        configuration.Inkling2TextConfig(num_hidden_layers=12, n_layers_per_attn_res_block=3, loop_config=loop_config)
    with pytest.raises(ValueError, match="normalize"):
        configuration.Inkling2TextConfig(
            num_hidden_layers=12,
            n_layers_per_attn_res_block=3,
            attn_res_normalize_values=False,
            loop_config=LOOPED["loop_config"],
        )


def test_variable_loop_checkpoint_trains_at_its_pass_count():
    text = configuration.Inkling2TextConfig(
        num_hidden_layers=12, n_layers_per_attn_res_block=3, loop_config={**LOOP, "is_varloop": True}
    )
    assert text.num_layer_visits == 12 + 2 * 6


@pytest.mark.parametrize(
    "model_type, text_config, ok",
    [
        ("inkling2_loop", LOOPED, True),
        ("inkling2", {"num_hidden_layers": 12}, True),
        ("inkling2", LOOPED, False),
        ("inkling2_loop", {"num_hidden_layers": 12}, False),
        # Pre-release checkpoints are not checked.
        ("inkling2_mm_model", LOOPED, True),
    ],
)
def test_model_type_matches_loop_config(tmp_path, model_type, text_config, ok):
    if ok:
        _load(tmp_path, {"model_type": model_type, "text_config": text_config})
    else:
        with pytest.raises(ValueError, match="inkling2_loop"):
            _load(tmp_path, {"model_type": model_type, "text_config": text_config})


def test_looped_config_serializes(tmp_path):
    config = _load(tmp_path, {"model_type": "inkling2_loop", "text_config": LOOPED})
    config.save_pretrained(tmp_path / "out")
    again = _load(tmp_path / "out", json.loads((tmp_path / "out" / "config.json").read_text()))
    assert again.model_type == "inkling2_loop" and again.get_text_config().loop_config == LOOP


def test_release_config_of_looped_pre_release_checkpoint():
    converted = configuration.release_config({"model_type": "inkling2_mm_model", "text_config": LOOPED})
    assert converted["model_type"] == "inkling2_loop"
