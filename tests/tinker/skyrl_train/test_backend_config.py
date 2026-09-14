from skyrl.backends.skyrl_train_backend import (
    MegatronBackendOverrides,
    _build_skyrl_train_config,
)


def test_api_model_fills_unspecified_trainer_paths():
    cfg = _build_skyrl_train_config("org/api-model", MegatronBackendOverrides())

    assert cfg.trainer.policy.model.path == "org/api-model"
    assert cfg.trainer.critic.model.path == "org/api-model"


def test_backend_config_preserves_explicit_trainer_paths():
    overrides = MegatronBackendOverrides(
        **{
            "trainer.policy.model.path": "/models/policy-revision",
            "trainer.critic.model.path": "/models/critic-revision",
        }
    )

    cfg = _build_skyrl_train_config("org/api-model", overrides)

    assert cfg.trainer.policy.model.path == "/models/policy-revision"
    assert cfg.trainer.critic.model.path == "/models/critic-revision"
