"""CPU unit tests for the MTP config knobs.

uv run --isolated --extra dev pytest tests/train/test_mtp_config.py
"""

from skyrl.train.config import InferenceEngineConfig, MegatronConfig
from skyrl.train.config.config import build_nested_dataclass


def test_megatron_config_mtp_defaults():
    cfg = MegatronConfig()
    # None => honor the model's own num_nextn_predict_layers (no SkyRL override).
    assert cfg.mtp_num_layers is None
    # Decoupled draft-training defaults.
    assert cfg.mtp_loss_weight == 0.1
    assert cfg.mtp_loss_type == "soft_ce"
    assert cfg.mtp_detach_trunk is True
    assert cfg.mtp_detach_shared_output is False
    # Deprecated alias is unset by default.
    assert cfg.mtp_loss_scaling_factor is None


def test_megatron_config_mtp_overrides_parse():
    cfg = build_nested_dataclass(
        MegatronConfig, {"mtp_num_layers": 2, "mtp_loss_weight": 0.3, "mtp_loss_type": "hard_ce"}
    )
    assert cfg.mtp_num_layers == 2
    assert cfg.mtp_loss_weight == 0.3
    assert cfg.mtp_loss_type == "hard_ce"


def test_megatron_config_mtp_loss_scaling_factor_back_compat():
    # The deprecated scaling-factor knob maps onto the explicit loss weight.
    cfg = build_nested_dataclass(MegatronConfig, {"mtp_loss_scaling_factor": 0.25})
    assert cfg.mtp_loss_weight == 0.25


def test_megatron_config_mtp_force_disable():
    # An explicit 0 is how a user force-disables MTP even on an MTP-capable model.
    cfg = build_nested_dataclass(MegatronConfig, {"mtp_num_layers": 0})
    assert cfg.mtp_num_layers == 0


def test_inference_engine_speculative_config_default_none():
    cfg = InferenceEngineConfig()
    assert cfg.speculative_config is None


def test_inference_engine_speculative_config_parses_mtp_dict():
    spec = {"method": "mtp", "num_speculative_tokens": 1}
    cfg = build_nested_dataclass(InferenceEngineConfig, {"speculative_config": spec})
    assert cfg.speculative_config == spec
