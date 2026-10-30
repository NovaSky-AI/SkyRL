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

from skyrl.backends import skyrl_train_backend
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
    """The request's flags land in the policy's SkyRLLoraConfig unchanged."""
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
    """The config `create_lora_training_client(rank=8)` sends, train_unembed=True included, must build."""
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
        """Adapters covering different groups have different shapes, so their signatures differ."""
        assert self._sig(train_mlp=False) != self._sig(train_attn=False)
        assert self._sig(train_mlp=False) != self._sig()

    def test_seed_is_excluded(self):
        """Seed changes initial values, not structure; including it would reject
        adapters that are perfectly compatible."""
        assert self._sig(seed=1) == self._sig(seed=2)

    def test_rank_and_alpha_still_count(self):
        """Rank and alpha remain part of the signature."""
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


class _StopBeforeWorkers(Exception):
    pass


def _build_critic_until_workers(monkeypatch, lora: types.LoraConfig):
    """Run _build_critic up to the point where it would start Ray workers."""
    backend = SkyRLTrainBackend.__new__(SkyRLTrainBackend)
    backend.config = FSDPBackendOverrides()  # _build_critic checks it against the request's rank
    backend._cfg = _build_skyrl_train_config(BASE_MODEL, FSDPBackendOverrides(), _lora())
    backend._colocate_pg = None

    def stop(*args, **kwargs):
        raise _StopBeforeWorkers

    monkeypatch.setattr(skyrl_train_backend, "PPORayActorGroup", stop)
    with pytest.raises(_StopBeforeWorkers):
        backend._build_critic(CriticWorker=None, lora_config=lora)
    return backend._cfg.trainer.critic.model.lora


def test_critic_receives_the_flags(monkeypatch):
    """_build_critic copies the request's rank and flags into the critic's LoRA config."""
    critic_lora = _build_critic_until_workers(monkeypatch, _lora(train_mlp=False))
    assert (critic_lora.rank, critic_lora.train_attn, critic_lora.train_mlp) == (8, True, False)


def test_critic_rejects_unappliable_flags_before_workers_start(monkeypatch):
    """An invalid critic request raises before any critic worker is created."""
    backend = SkyRLTrainBackend.__new__(SkyRLTrainBackend)
    backend.config = FSDPBackendOverrides()
    backend._cfg = _build_skyrl_train_config(BASE_MODEL, FSDPBackendOverrides(), _lora())
    backend._cfg.trainer.critic.model.lora.target_modules = ["q_proj"]
    monkeypatch.setattr(skyrl_train_backend, "PPORayActorGroup", lambda *a, **k: pytest.fail("workers started"))
    with pytest.raises(ValueError, match="narrow"):
        backend._build_critic(CriticWorker=None, lora_config=_lora(train_mlp=False))


@pytest.mark.parametrize(
    "key",
    ["trainer.policy.model.lora.rank", "trainer.policy.model.lora.train_mlp", "trainer.critic.model.lora.train_attn"],
)
def test_backend_config_may_not_set_request_fields(key):
    """The request always carries these, so a backend_config value could never take effect."""
    with pytest.raises(ValueError, match="may not set"):
        _build_skyrl_train_config(BASE_MODEL, FSDPBackendOverrides(**{key: False}), _lora())


def test_backend_config_alpha_still_overrides_the_api_placeholder():
    """An operator's alpha in backend_config wins over the API's placeholder of 32."""
    overrides = FSDPBackendOverrides(**{"trainer.policy.model.lora.alpha": 64})
    cfg = _build_skyrl_train_config(BASE_MODEL, overrides, _lora())
    assert cfg.trainer.policy.model.lora.alpha == 64


def test_backend_config_rank_is_allowed_without_a_request():
    """An inference-only server builds its config with no request, so backend_config is its only source."""
    overrides = FSDPBackendOverrides(**{"trainer.policy.model.lora.rank": 16})
    cfg = _build_skyrl_train_config(BASE_MODEL, overrides, None)
    assert cfg.trainer.policy.model.lora.rank == 16
