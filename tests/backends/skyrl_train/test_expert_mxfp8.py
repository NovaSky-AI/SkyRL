from argparse import Namespace
from fnmatch import fnmatch

import pytest

from skyrl.backends.skyrl_train.inference_servers.utils import (
    apply_expert_mxfp8_rollout_config,
)
from skyrl.backends.skyrl_train.quantization import Mxfp8Strategy, get_model_quantization_policy
from skyrl.backends.skyrl_train.quantization.megatron import build_megatron_quantization_recipe
from skyrl.train.config import SkyRLTrainConfig


def _enabled_config() -> SkyRLTrainConfig:
    return SkyRLTrainConfig.from_cli_overrides(
        [
            "trainer.strategy=megatron",
            "trainer.policy.model.expert_mxfp8.enabled=true",
        ]
    )


def _recipe(*, persistent: bool = False) -> dict:
    policy = get_model_quantization_policy("qwen3_moe", Mxfp8Strategy.quantized_categories)
    return build_megatron_quantization_recipe(policy, format_name="mxfp8", persistent=persistent)


def test_expert_recipe_targets_only_routed_experts():
    recipe = _recipe()
    fc1_pattern = recipe["matchers"]["routed_fc1"]["pattern"]
    fc2_pattern = recipe["matchers"]["routed_fc2"]["pattern"]
    assert fnmatch("decoder.layers.0.mlp.experts.linear_fc1", fc1_pattern)
    assert fnmatch("decoder.layers.0.mlp.experts.linear_fc2", fc2_pattern)
    assert not fnmatch("decoder.layers.0.mlp.shared_experts.linear_fc1", fc1_pattern)
    assert not fnmatch("decoder.layers.0.mlp.shared_experts.linear_fc2", fc2_pattern)


def test_persistent_expert_recipe_enables_fp8_params_only_for_routed_experts():
    default_recipe = _recipe()
    persistent_recipe = _recipe(persistent=True)

    for phase in ("training_recipe", "evaluation_recipe"):
        assert "fp8_param" not in default_recipe["configs"]["quantized"][phase]
        assert persistent_recipe["configs"]["quantized"][phase]["fp8_param"] is True
        assert "fp8_param" not in persistent_recipe["configs"]["high_precision"][phase]


def test_rollout_config_enables_expert_only_mxfp8():
    args = Namespace()
    apply_expert_mxfp8_rollout_config(args, _enabled_config(), {})
    assert args.quantization == "online"
    assert args.quantization_config == {"moe": "mxfp8"}


def test_rollout_config_preserves_serialized_mxfp8_settings():
    cfg = _enabled_config()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "serialized_mxfp8"
    args = Namespace()

    apply_expert_mxfp8_rollout_config(args, cfg, {"quantization": "modelopt_mxfp8"})

    assert not hasattr(args, "quantization")
    assert not hasattr(args, "quantization_config")


def test_rollout_config_rejects_float32():
    cfg = _enabled_config()
    cfg.generator.inference_engine.model_dtype = "float32"
    with pytest.raises(ValueError, match="model_dtype"):
        apply_expert_mxfp8_rollout_config(Namespace(), cfg, {})


@pytest.mark.parametrize("key", ["quantization", "quantization_config"])
def test_rollout_config_rejects_quantization_overrides(key):
    with pytest.raises(ValueError, match="engine_init_kwargs"):
        apply_expert_mxfp8_rollout_config(Namespace(), _enabled_config(), {key: "conflict"})
