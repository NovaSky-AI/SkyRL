"""CPU unit tests for the MTP config knobs.

uv run --isolated --extra dev pytest tests/train/test_mtp_config.py
"""

import pytest

from skyrl.train.config import (
    InferenceEngineConfig,
    MegatronConfig,
    MTPConfig,
    SkyRLTrainConfig,
)
from skyrl.train.config.config import build_nested_dataclass
from skyrl.train.utils.utils import _apply_mtp_config, _validate_draft_weight_sync_cfg


def test_megatron_config_mtp_defaults():
    cfg = MegatronConfig()
    # None => honor the model's own num_nextn_predict_layers (no SkyRL override).
    assert cfg.mtp_num_layers is None
    # Decoupled draft-training defaults. The decoupling itself is unconditional (no knob): the draft
    # loss trains only the MTP-head parameters -- trunk, teacher, output projection and the MTP
    # block's re-embedding are all detached (see mtp/hidden_capture.py, mtp/adapter.py).
    assert cfg.mtp_loss_weight == 0.1
    assert cfg.mtp_loss_topk is None


def test_megatron_config_mtp_overrides_parse():
    cfg = build_nested_dataclass(MegatronConfig, {"mtp_num_layers": 2, "mtp_loss_weight": 0.3, "mtp_loss_topk": 64})
    assert cfg.mtp_num_layers == 2
    assert cfg.mtp_loss_weight == 0.3
    assert cfg.mtp_loss_topk == 64


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


def test_mtp_config_defaults():
    cfg = MTPConfig()
    assert cfg.enabled is False
    assert cfg.num_speculative_tokens == 1
    assert cfg.loss_weight == 0.1


def test_apply_mtp_config_enabled_propagates_to_training_and_inference():
    cfg = SkyRLTrainConfig()
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.num_speculative_tokens = 2
    cfg.trainer.mtp.loss_weight = 0.25
    _apply_mtp_config(cfg)
    # Draft depth is inference-only: the trained head count stays None (=> the bridge infers it
    # from the checkpoint), so num_speculative_tokens > 1 reuses the single head autoregressively
    # in vLLM instead of force-building extra randomly-initialized Megatron heads.
    assert cfg.trainer.policy.megatron_config.mtp_num_layers is None
    assert cfg.trainer.policy.megatron_config.mtp_loss_weight == 0.25
    assert cfg.generator.inference_engine.speculative_config == {
        "method": "mtp",
        "num_speculative_tokens": 2,
    }


def test_apply_mtp_config_keeps_explicit_head_override():
    # A user can still pin the trained head count (e.g. force-build fresh heads on a model that
    # ships without them); the draft depth stays independent.
    cfg = SkyRLTrainConfig()
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.num_speculative_tokens = 3
    cfg.trainer.policy.megatron_config.mtp_num_layers = 1
    _apply_mtp_config(cfg)
    assert cfg.trainer.policy.megatron_config.mtp_num_layers == 1
    assert cfg.generator.inference_engine.speculative_config["num_speculative_tokens"] == 3


def test_apply_mtp_config_rejects_enabled_with_zero_heads():
    # mtp_num_layers=0 means "force-disable MTP" — contradicts trainer.mtp.enabled=true.

    cfg = SkyRLTrainConfig()
    cfg.trainer.mtp.enabled = True
    cfg.trainer.policy.megatron_config.mtp_num_layers = 0
    with pytest.raises(ValueError, match="mtp_num_layers=0"):
        _apply_mtp_config(cfg)


def test_apply_mtp_config_disabled_force_disables_heads():
    cfg = SkyRLTrainConfig()
    _apply_mtp_config(cfg)
    assert cfg.trainer.policy.megatron_config.mtp_num_layers == 0
    assert cfg.generator.inference_engine.speculative_config is None


def test_apply_mtp_config_does_not_clobber_explicit_speculative_config():
    cfg = SkyRLTrainConfig()
    cfg.trainer.mtp.enabled = True
    cfg.generator.inference_engine.speculative_config = {"method": "mtp", "num_speculative_tokens": 5}
    _apply_mtp_config(cfg)
    assert cfg.generator.inference_engine.speculative_config["num_speculative_tokens"] == 5


def _spec_cfg(strategy="megatron", weight_sync_backend="nccl"):
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = strategy
    cfg.trainer.mtp.enabled = True
    cfg.generator.inference_engine.weight_sync_backend = weight_sync_backend
    _apply_mtp_config(cfg)
    assert cfg.generator.inference_engine.speculative_config["method"] == "mtp"
    return cfg


def test_draft_weight_sync_cfg_accepts_megatron_nccl():
    _validate_draft_weight_sync_cfg(_spec_cfg())


def test_draft_weight_sync_cfg_noop_without_spec_decode():
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fsdp"
    _validate_draft_weight_sync_cfg(cfg)
    cfg.generator.inference_engine.speculative_config = {"method": "ngram", "prompt_lookup_max": 4}
    _validate_draft_weight_sync_cfg(cfg)


@pytest.mark.parametrize(
    "speculative_config",
    [
        {"method": "eagle3", "model": "some/eagle-head"},
        {"method": "dflash", "model": "some/dflash-head"},
        {"model": "some/external-draft-checkpoint"},
    ],
)
def test_draft_weight_sync_cfg_leaves_external_draft_models_unchanged(speculative_config):
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fsdp"
    cfg.generator.inference_engine.speculative_config = speculative_config
    _validate_draft_weight_sync_cfg(cfg)


def test_draft_weight_sync_cfg_rejects_fsdp():
    with pytest.raises(ValueError, match="requires trainer.strategy='megatron'"):
        _validate_draft_weight_sync_cfg(_spec_cfg(strategy="fsdp"))


@pytest.mark.parametrize("backend", ["delta", "sharded_rdt"])
def test_draft_weight_sync_cfg_rejects_non_retargeting_backends(backend):
    with pytest.raises(ValueError, match=f"weight_sync_backend={backend!r}"):
        _validate_draft_weight_sync_cfg(_spec_cfg(weight_sync_backend=backend))


def test_draft_weight_sync_cfg_rejects_fp8_weight_sync():
    cfg = _spec_cfg()
    cfg.generator.inference_engine.fp8_weight_sync_mode = "blockwise"
    with pytest.raises(ValueError, match="fp8_weight_sync_mode='blockwise'"):
        _validate_draft_weight_sync_cfg(cfg)


def test_draft_weight_sync_cfg_rejects_adapter_only_lora():
    from skyrl.train.config import SkyRLLoraConfig

    cfg = _spec_cfg()
    cfg.trainer.policy.model.lora = SkyRLLoraConfig(rank=16, alpha=16)
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = False
    with pytest.raises(ValueError, match="full-weight sync"):
        _validate_draft_weight_sync_cfg(cfg)
    cfg.trainer.policy.megatron_config.lora_config.merge_lora = True
    _validate_draft_weight_sync_cfg(cfg)
