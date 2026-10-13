"""CPU tests for Tinker's train_attn / train_mlp on skyrl-train.

Covers the request -> SkyRLTrainConfig mapping and the multi-adapter signature.
The worker-side narrowing these feed into is tested separately in
tests/backends/skyrl_train/utils/test_lora_targets.py.

Lives under ``skyrl_train`` because it reaches into the SkyRL-Train config, so it
needs a backend extra. No Ray cluster and no GPU. Run:
  uv run --isolated --extra fsdp --extra tinker --extra dev \\
    pytest tests/tinker/skyrl_train/test_lora_flags.py
"""

import pytest

from skyrl.backends.skyrl_train_backend import (
    FSDPBackendOverrides,
    MegatronBackendOverrides,
    SkyRLTrainBackend,
    _build_skyrl_train_config,
)
from skyrl.tinker import types

BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _lora(**kwargs) -> types.LoraConfig:
    return types.LoraConfig(**{"rank": 8, "alpha": 32.0, "seed": 0, **kwargs})


@pytest.mark.parametrize(
    ("train_attn", "train_mlp"),
    [(True, True), (True, False), (False, True)],
)
def test_flags_reach_the_policy_lora_config(train_attn, train_mlp):
    cfg = _build_skyrl_train_config(
        BASE_MODEL,
        FSDPBackendOverrides(),
        _lora(train_attn=train_attn, train_mlp=train_mlp),
    )
    assert cfg.trainer.policy.model.lora.train_attn is train_attn
    assert cfg.trainer.policy.model.lora.train_mlp is train_mlp


def test_defaults_adapt_both_groups():
    """A client that sends no flags must keep the pre-existing behaviour."""
    cfg = _build_skyrl_train_config(BASE_MODEL, FSDPBackendOverrides(), _lora())
    assert cfg.trainer.policy.model.lora.train_attn is True
    assert cfg.trainer.policy.model.lora.train_mlp is True
    assert cfg.trainer.policy.model.lora.target_modules == "all-linear"


def test_sdk_default_lora_config_is_accepted():
    """The exact config `create_lora_training_client(rank=8)` produces must work.

    Regression guard: an earlier revision rejected train_unembed=True, which is
    the SDK default, and broke every caller that passed no flags at all.
    """
    import tinker

    sdk_cfg = tinker.LoraConfig(rank=8)
    lora = types.LoraConfig(
        rank=sdk_cfg.rank,
        alpha=32.0,
        seed=0,
        train_attn=sdk_cfg.train_attn,
        train_mlp=sdk_cfg.train_mlp,
        train_unembed=sdk_cfg.train_unembed,
    )
    cfg = _build_skyrl_train_config(BASE_MODEL, FSDPBackendOverrides(), lora)
    assert cfg.trainer.policy.model.lora.train_attn is True
    assert cfg.trainer.policy.model.lora.train_mlp is True


def test_flags_are_not_applied_without_lora():
    """rank=0 means no adapter at all; the flags must not leak into the config."""
    cfg = _build_skyrl_train_config(BASE_MODEL, FSDPBackendOverrides(), _lora(rank=0, train_mlp=False))
    assert cfg.trainer.policy.model.lora.rank == 0
    assert cfg.trainer.policy.model.lora.train_mlp is True


class TestLoraSignature:
    """Adapters sharing a runtime are byte-copies of one template, so the
    signature must cover everything that changes an adapter's *shape*."""

    @staticmethod
    def _sig(**kwargs) -> tuple:
        return SkyRLTrainBackend._lora_signature_from(None, _lora(**kwargs))

    def test_differing_groups_produce_different_signatures(self):
        assert self._sig(train_mlp=False) != self._sig(train_attn=False)
        assert self._sig(train_mlp=False) != self._sig()

    def test_seed_is_excluded(self):
        """Seed changes initial values, not structure; including it would reject
        adapters that are perfectly compatible."""
        assert self._sig(seed=1) == self._sig(seed=2)

    def test_rank_and_alpha_still_count(self):
        assert self._sig(rank=16) != self._sig(rank=8)
        assert self._sig(alpha=16.0) != self._sig(alpha=32.0)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (FSDPBackendOverrides(**{"trainer.policy.model.lora.target_modules": ["q_proj"]}), ValueError),
        (MegatronBackendOverrides(), NotImplementedError),
    ],
    ids=["explicit_target_modules", "megatron"],
)
def test_unappliable_flags_fail_while_building_the_config(overrides, error):
    """Raised here, before create_model starts Ray or reserves GPUs."""
    with pytest.raises(error):
        _build_skyrl_train_config(BASE_MODEL, overrides, _lora(train_mlp=False))
