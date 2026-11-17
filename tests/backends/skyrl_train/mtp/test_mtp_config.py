"""CPU unit tests for the MTP config knobs.

uv run --isolated --extra dev pytest tests/train/test_mtp_config.py
"""

import pytest
from loguru import logger
from omegaconf import OmegaConf

from skyrl.train.config import (
    InferenceEngineConfig,
    MegatronConfig,
    MTPConfig,
    SkyRLTrainConfig,
    get_config_as_yaml_str,
)
from skyrl.train.config.config import MTPLossType, build_nested_dataclass
from skyrl.train.utils.utils import (
    _apply_mtp_config,
    _validate_draft_weight_sync_cfg,
    validate_cfg,
    validate_megatron_cfg,
)
from tests.train.util import example_dummy_config


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


def _spec_cfg(strategy="megatron", weight_sync_backend="nccl", colocate_all=False):
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = strategy
    cfg.trainer.mtp.enabled = True
    cfg.trainer.placement.colocate_all = colocate_all
    cfg.generator.inference_engine.weight_sync_backend = weight_sync_backend
    _apply_mtp_config(cfg)
    assert cfg.generator.inference_engine.speculative_config["method"] == "mtp"
    return cfg


@pytest.mark.parametrize(
    ("weight_sync_backend", "colocate_all"),
    [("nccl", False), ("nccl", True), ("delta", False), ("sharded_rdt", False), ("rdt", False)],
)
def test_draft_weight_sync_cfg_accepts_megatron_full_weight_backends(weight_sync_backend, colocate_all):
    _validate_draft_weight_sync_cfg(_spec_cfg(weight_sync_backend=weight_sync_backend, colocate_all=colocate_all))


def test_draft_weight_sync_cfg_noop_without_spec_decode():
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fsdp"
    cfg.generator.inference_engine.weight_sync_backend = "sharded_rdt"
    _validate_draft_weight_sync_cfg(cfg)


def test_draft_weight_sync_cfg_rejects_fsdp():
    with pytest.raises(ValueError, match="requires trainer.strategy='megatron'"):
        _validate_draft_weight_sync_cfg(_spec_cfg(strategy="fsdp"))


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


def test_mtp_loss_type_defaults_to_hard_label_and_propagates():
    assert MTPConfig().loss_type == MegatronConfig().mtp_loss_type == MTPLossType.HARD_LABEL
    cfg = SkyRLTrainConfig()
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.loss_type = "distribution"
    _apply_mtp_config(cfg)
    assert cfg.trainer.policy.megatron_config.mtp_loss_type == "distribution"


def test_mtp_loss_type_parses_from_yaml_and_cli():
    # YAML and CLI overrides arrive as plain strings and stay strings, so these fields dump as plain YAML.
    assert MTPConfig.from_dict_config(OmegaConf.create("loss_type: distribution")).loss_type == MTPLossType.DISTRIBUTION
    cfg = SkyRLTrainConfig.from_cli_overrides(["trainer.mtp.enabled=true", "trainer.mtp.loss_type=distribution"])
    _apply_mtp_config(cfg)
    assert cfg.trainer.policy.megatron_config.mtp_loss_type == MTPLossType.DISTRIBUTION
    dumped = [line.strip() for line in get_config_as_yaml_str(cfg).splitlines()]
    assert "loss_type: distribution" in dumped and "mtp_loss_type: distribution" in dumped
    with pytest.raises(ValueError, match="trainer.mtp.loss_type must be one of"):
        SkyRLTrainConfig.from_cli_overrides(["trainer.mtp.loss_type=hard-label"])


@pytest.mark.parametrize(
    ("loss_type", "topk", "warns"),
    [("hard_label", 256, True), ("hard_label", None, False), ("distribution", 256, False)],
)
def test_apply_mtp_config_warns_hard_label_ignores_topk(loss_type, topk, warns):
    cfg = SkyRLTrainConfig()
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.loss_type = loss_type
    cfg.trainer.policy.megatron_config.mtp_loss_topk = topk
    messages = []
    handler_id = logger.add(messages.append, level="WARNING")
    try:
        _apply_mtp_config(cfg)
    finally:
        logger.remove(handler_id)
    assert any("mtp_loss_topk is ignored" in m for m in messages) is warns


def _megatron_mtp_cfg(loss_type, *, cp=1, pp=1, fused_lm_head=False):
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.fused_lm_head_logprob = fused_lm_head
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.loss_type = loss_type
    cfg.trainer.policy.megatron_config.context_parallel_size = cp
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    return cfg


@pytest.mark.parametrize(
    ("loss_type", "cp", "pp", "fused_lm_head"),
    [
        ("hard_label", 2, 1, False),
        ("hard_label", 1, 2, False),
        ("hard_label", 1, 1, True),
        ("hard_label", 2, 2, True),
        ("distribution", 1, 1, False),
    ],
)
def test_validate_megatron_cfg_accepts_supported_mtp_parallelism(loss_type, cp, pp, fused_lm_head):
    cfg = _megatron_mtp_cfg(loss_type, cp=cp, pp=pp, fused_lm_head=fused_lm_head)
    _apply_mtp_config(cfg)
    validate_megatron_cfg(cfg)


@pytest.mark.parametrize(
    ("cp", "pp", "fused_lm_head", "match"),
    [
        (2, 1, False, "context_parallel_size > 1"),
        (1, 2, False, "pipeline_model_parallel_size > 1"),
        (1, 1, True, "trainer.fused_lm_head_logprob=true"),
    ],
)
def test_validate_megatron_cfg_refuses_distribution_mtp_with_cp_pp_or_fused_head(cp, pp, fused_lm_head, match):
    cfg = _megatron_mtp_cfg(MTPLossType.DISTRIBUTION, cp=cp, pp=pp, fused_lm_head=fused_lm_head)
    with pytest.raises(ValueError, match=f"'distribution' does not support {match}.*loss_type='hard_label'"):
        validate_megatron_cfg(cfg)


def test_validate_megatron_cfg_refuses_hard_label_mtp_cp_without_packing():
    cfg = _megatron_mtp_cfg(MTPLossType.HARD_LABEL, cp=2)
    cfg.trainer.remove_microbatch_padding = False
    with pytest.raises(AssertionError, match="context parallel is only supported with remove_microbatch_padding"):
        validate_megatron_cfg(cfg)


def test_validate_cfg_accepts_hard_label_mtp_with_cp():
    # validate_cfg checks the Megatron config before _apply_mtp_config copies trainer.mtp.loss_type into it.
    cfg = example_dummy_config()
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.critic_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.strategy = "megatron"
    cfg.trainer.remove_microbatch_padding = True
    cfg.trainer.mtp.enabled = True
    cfg.trainer.mtp.loss_type = MTPLossType.HARD_LABEL
    cfg.trainer.policy.megatron_config.context_parallel_size = 2
    # CP=2 needs two policy GPUs; colocate_all pairs them with two rollout GPUs.
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.generator.inference_engine.num_engines = 2
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    validate_cfg(cfg)
    assert cfg.trainer.policy.megatron_config.mtp_loss_type == MTPLossType.HARD_LABEL


def test_mtp_lr_validation():
    def apply(enabled, lr):
        cfg = SkyRLTrainConfig()
        cfg.trainer.mtp.enabled, cfg.trainer.mtp.lr = enabled, lr
        cfg.trainer.mtp.loss_type = "hard_label"
        messages = []
        handler_id = logger.add(messages.append, level="WARNING")
        try:
            _apply_mtp_config(cfg)
        finally:
            logger.remove(handler_id)
        return messages

    # MTP off: the lr is ignored, with a warning.
    assert any("ignored" in m for m in apply(False, 1e-5))
    # No lr: the heads train at the policy's, with a warning.
    assert any("policy's learning rate" in m for m in apply(True, None))
    assert not apply(True, 1e-5)
    # A non-positive lr is rejected when the config is built, whether or not MTP is enabled.
    for enabled in (True, False):
        with pytest.raises(ValueError, match="must be > 0"):
            MTPConfig(enabled=enabled, lr=0.0)
