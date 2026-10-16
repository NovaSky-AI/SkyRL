"""CPU unit tests for the MTP config knobs.

uv run --isolated --extra dev --extra skyrl-train pytest tests/backends/skyrl_train/mtp/test_mtp_config.py
"""

import pytest

from skyrl.train.config import (
    InferenceEngineConfig,
    MegatronConfig,
    MTPConfig,
    SkyRLTrainConfig,
)
from skyrl.train.config.config import build_nested_dataclass
from skyrl.train.utils.utils import (
    _apply_mtp_config,
    _validate_draft_weight_sync_cfg,
    _validate_mtp_prefix_caching,
    validate_cfg,
    validate_inference_engine_cfg,
)


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


@pytest.mark.parametrize(
    ("enable_prefix_caching", "engine_init_kwargs", "should_raise"),
    [
        (True, {}, True),
        (False, {}, False),
        (False, {"enable_prefix_caching": True}, True),
        (True, {"enable_prefix_caching": False}, False),
    ],
)
def test_validate_mtp_prefix_caching_for_linear_attention_model(
    tmp_path, enable_prefix_caching, engine_init_kwargs, should_raise
):
    from transformers import Qwen3_5TextConfig

    Qwen3_5TextConfig(
        num_hidden_layers=4,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    ).save_pretrained(tmp_path)

    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = str(tmp_path)
    cfg.trainer.mtp.enabled = True
    cfg.generator.inference_engine.enable_prefix_caching = enable_prefix_caching
    cfg.generator.inference_engine.engine_init_kwargs = engine_init_kwargs
    _apply_mtp_config(cfg)

    if should_raise:
        with pytest.raises(ValueError, match="enable_prefix_caching=false"):
            _validate_mtp_prefix_caching(cfg)
    else:
        _validate_mtp_prefix_caching(cfg)


def test_validate_mtp_prefix_caching_allows_full_attention_model(tmp_path):
    from transformers import Qwen3_5TextConfig

    Qwen3_5TextConfig(num_hidden_layers=2, layer_types=["full_attention", "full_attention"]).save_pretrained(tmp_path)

    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = str(tmp_path)
    cfg.trainer.mtp.enabled = True
    _apply_mtp_config(cfg)

    _validate_mtp_prefix_caching(cfg)


def test_validate_mtp_prefix_caching_allows_effective_non_mtp_override():
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = "unresolved-linear-attention-model"
    cfg.trainer.mtp.enabled = True
    cfg.generator.inference_engine.engine_init_kwargs = {
        "speculative_config": {
            "method": "eagle",
            "num_speculative_tokens": 1,
        }
    }
    _apply_mtp_config(cfg)

    _validate_mtp_prefix_caching(cfg)


def test_validate_mtp_prefix_caching_rejects_implicit_native_mtp(tmp_path):
    from transformers import Qwen3_5Config, Qwen3_5TextConfig

    model_path = tmp_path / "model"
    Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=Qwen3_5TextConfig(
            num_hidden_layers=2,
            layer_types=["linear_attention", "full_attention"],
            mtp_num_hidden_layers=1,
        ),
    ).save_pretrained(model_path)

    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = str(model_path)
    cfg.generator.inference_engine.speculative_config = {
        "model": str(model_path),
        "num_speculative_tokens": 1,
    }

    with pytest.raises(ValueError, match="MTP speculative decoding with prefix caching"):
        _validate_mtp_prefix_caching(cfg)


@pytest.mark.parametrize("model", ["draft", "ngram", "package.CustomProposer"])
def test_validate_mtp_prefix_caching_allows_implicit_non_mtp_model(tmp_path, model):
    from transformers import GPT2Config, Qwen3_5TextConfig

    target_path = tmp_path / "target"
    draft_path = tmp_path / "draft"
    Qwen3_5TextConfig(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    ).save_pretrained(target_path)
    GPT2Config().save_pretrained(draft_path)

    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = str(target_path)
    cfg.generator.inference_engine.speculative_config = {
        "model": str(draft_path) if model == "draft" else model,
        "num_speculative_tokens": 1,
    }

    _validate_mtp_prefix_caching(cfg)


def test_validate_cfg_rejects_mtp_prefix_caching_after_mtp_propagation(tmp_path):
    from transformers import Qwen3_5TextConfig

    Qwen3_5TextConfig(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    ).save_pretrained(tmp_path)

    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = str(tmp_path)
    cfg.trainer.strategy = "megatron"
    cfg.trainer.mtp.enabled = True
    cfg.trainer.logger = "console"

    with pytest.raises(ValueError, match="MTP speculative decoding with prefix caching"):
        validate_cfg(cfg)

    assert cfg.generator.inference_engine.speculative_config == {
        "method": "mtp",
        "num_speculative_tokens": 1,
    }


@pytest.mark.parametrize("enable_prefix_caching", [False, True])
def test_inference_only_validation_checks_mtp_prefix_caching(tmp_path, enable_prefix_caching):
    from transformers import Qwen3_5TextConfig

    Qwen3_5TextConfig(
        num_hidden_layers=2,
        layer_types=["linear_attention", "full_attention"],
    ).save_pretrained(tmp_path)

    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = str(tmp_path)
    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.enable_prefix_caching = enable_prefix_caching
    cfg.generator.inference_engine.speculative_config = {"method": "mtp", "num_speculative_tokens": 1}

    # The serve entrypoint calls this shared validator without training's MTP propagation.
    if enable_prefix_caching:
        with pytest.raises(ValueError, match="MTP speculative decoding with prefix caching"):
            validate_inference_engine_cfg(cfg)
    else:
        validate_inference_engine_cfg(cfg)


@pytest.mark.parametrize("hybrid_override", [False, True])
def test_validate_mtp_prefix_caching_uses_effective_engine_model(tmp_path, hybrid_override):
    from transformers import Qwen3_5TextConfig

    full_attention_path = tmp_path / "full_attention"
    hybrid_path = tmp_path / "hybrid"
    Qwen3_5TextConfig(num_hidden_layers=2, layer_types=["full_attention", "full_attention"]).save_pretrained(
        full_attention_path
    )
    Qwen3_5TextConfig(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"]).save_pretrained(
        hybrid_path
    )

    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = str(full_attention_path if hybrid_override else hybrid_path)
    cfg.generator.inference_engine.engine_init_kwargs = {
        "model": str(hybrid_path if hybrid_override else full_attention_path),
    }
    cfg.generator.inference_engine.speculative_config = {"method": "mtp", "num_speculative_tokens": 1}

    if hybrid_override:
        with pytest.raises(ValueError, match="MTP speculative decoding with prefix caching"):
            validate_inference_engine_cfg(cfg)
    else:
        validate_inference_engine_cfg(cfg)


@pytest.mark.parametrize(
    ("decode_init_kwargs", "should_raise"),
    [
        ({"enable_prefix_caching": True}, True),
        ({"enable_prefix_caching": False}, False),
        ({"speculative_config": {"method": "mtp", "num_speculative_tokens": 1}}, True),
        ({"speculative_config": {"method": "eagle", "num_speculative_tokens": 1}}, False),
    ],
)
def test_validate_mtp_prefix_caching_checks_pd_role_kwargs(tmp_path, decode_init_kwargs, should_raise):
    from transformers import Qwen3_5TextConfig

    Qwen3_5TextConfig(num_hidden_layers=2, layer_types=["linear_attention", "full_attention"]).save_pretrained(tmp_path)

    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = str(tmp_path)
    cfg.generator.inference_engine.enable_pd = True
    cfg.generator.inference_engine.enable_prefix_caching = decode_init_kwargs.get("enable_prefix_caching", True)
    if "speculative_config" not in decode_init_kwargs:
        cfg.generator.inference_engine.speculative_config = {"method": "mtp", "num_speculative_tokens": 1}

    cfg.generator.inference_engine.prefill_init_kwargs = {
        "enable_prefix_caching": False,
        "kv_transfer_config": {"kv_connector": "PyTorchConnector"},
    }
    decode_kwargs = {"kv_transfer_config": {"kv_connector": "PyTorchConnector"}}
    decode_kwargs.update(decode_init_kwargs)
    cfg.generator.inference_engine.decode_init_kwargs = decode_kwargs

    if should_raise:
        with pytest.raises(ValueError, match="MTP speculative decoding with prefix caching"):
            _validate_mtp_prefix_caching(cfg)
    else:
        _validate_mtp_prefix_caching(cfg)


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
    [("nccl", False), ("nccl", True), ("delta", False)],
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


@pytest.mark.parametrize("backend", ["sharded_rdt", "rdt"])
def test_draft_weight_sync_cfg_rejects_sharded_rdt(backend):
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
