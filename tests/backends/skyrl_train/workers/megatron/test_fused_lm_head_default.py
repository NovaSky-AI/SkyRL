"""How ``trainer.fused_lm_head_logprob=None`` resolves in the Megatron worker."""

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.megatron


def _cfg(setting, lora_rank=0, target_modules="all-linear"):
    return SimpleNamespace(
        fused_lm_head_logprob=setting,
        policy=SimpleNamespace(
            model=SimpleNamespace(lora=SimpleNamespace(rank=lora_rank, target_modules=target_modules))
        ),
    )


@pytest.fixture
def resolve(monkeypatch):
    """Call ``MegatronModelWrapper._resolve_fused_lm_head`` with a stubbed model check."""
    wrapper_mod = pytest.importorskip("skyrl.backends.skyrl_train.workers.megatron.megatron_model_wrapper")
    warnings = []
    monkeypatch.setattr(wrapper_mod.logger, "warning", warnings.append)

    def _resolve(cfg, reason=None, mtp_num_layers=None):
        calls = []

        def unsupported_reason(model):
            calls.append(model)
            return reason

        monkeypatch.setattr(wrapper_mod, "fused_lm_head_unsupported_reason", unsupported_reason)
        monkeypatch.setattr(
            wrapper_mod, "get_model_config", lambda model: SimpleNamespace(mtp_num_layers=mtp_num_layers)
        )
        fake_self = SimpleNamespace(cfg=cfg, actor_module=["model"])
        return wrapper_mod.MegatronModelWrapper._resolve_fused_lm_head(fake_self), calls

    _resolve.warnings = warnings
    return _resolve


@pytest.mark.parametrize("setting", [True, False])
def test_explicit_setting_is_used_without_checking_the_model(resolve, setting):
    resolved, calls = resolve(_cfg(setting), reason="unsupported", mtp_num_layers=1)
    assert resolved is setting
    assert calls == []
    assert resolve.warnings == []


def test_default_enables_the_fused_head_on_a_supported_model(resolve):
    resolved, calls = resolve(_cfg(None))
    assert resolved is True
    assert calls == ["model"]
    assert resolve.warnings == []


@pytest.mark.parametrize(
    "cfg,reason,mtp_num_layers,expected",
    [
        (_cfg(None), "fused_lm_head_logprob does not support MuP logit scaling", None, "MuP"),
        (_cfg(None), None, 1, "MTP draft training"),
        (_cfg(None, lora_rank=8, target_modules=["linear_qkv", "output_layer"]), None, None, "output_layer"),
    ],
    ids=["unsupported_model", "mtp", "lora_on_output_layer"],
)
def test_default_falls_back_to_the_unfused_head_with_a_warning(resolve, cfg, reason, mtp_num_layers, expected):
    resolved, _ = resolve(cfg, reason=reason, mtp_num_layers=mtp_num_layers)
    assert resolved is False
    assert len(resolve.warnings) == 1
    assert "using the unfused LM head" in resolve.warnings[0]
    assert expected in resolve.warnings[0]


def test_lora_without_output_layer_keeps_the_fused_head(resolve):
    resolved, _ = resolve(_cfg(None, lora_rank=8, target_modules=["linear_qkv", "linear_proj"]))
    assert resolved is True


@pytest.fixture
def init_adapter_check(monkeypatch):
    worker_mod = pytest.importorskip("skyrl.backends.skyrl_train.workers.megatron.megatron_worker")
    warnings = []
    monkeypatch.setattr(worker_mod.logger, "warning", warnings.append)
    worker_mod._resolve_fused_lm_head_for_init_adapter.warnings = warnings
    return worker_mod._resolve_fused_lm_head_for_init_adapter


def test_init_adapter_on_lm_head_turns_the_default_off(init_adapter_check):
    cfg = SimpleNamespace(fused_lm_head_logprob=None)
    init_adapter_check(cfg, ["linear_qkv", "output_layer"])
    assert cfg.fused_lm_head_logprob is False
    assert "init_adapter_path trains lm_head" in init_adapter_check.warnings[0]


def test_init_adapter_on_lm_head_rejects_an_explicit_true(init_adapter_check):
    with pytest.raises(ValueError, match="adapter trains lm_head"):
        init_adapter_check(SimpleNamespace(fused_lm_head_logprob=True), ["output_layer"])


@pytest.mark.parametrize("setting", [None, True, False])
def test_init_adapter_without_lm_head_leaves_the_setting_alone(init_adapter_check, setting):
    cfg = SimpleNamespace(fused_lm_head_logprob=setting)
    init_adapter_check(cfg, ["linear_qkv", "linear_proj"])
    assert cfg.fused_lm_head_logprob is setting
    assert init_adapter_check.warnings == []
