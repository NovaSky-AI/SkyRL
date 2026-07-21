import pytest

from skyrl.train.config import SkyRLTrainConfig, get_config_as_yaml_str
from skyrl.train.utils.utils import validate_cfg


def _valid_fireworks_cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fireworks"
    cfg.trainer.fireworks.base_model = "accounts/fireworks/models/test"
    cfg.trainer.fireworks.max_seq_len = 4096
    cfg.trainer.policy.model.path = "Test/Tokenizer"
    cfg.trainer.policy.model.lora.rank = 8
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 0
    cfg.trainer.policy.optimizer_config.scheduler = "constant_with_warmup"
    cfg.trainer.algorithm.advantage_estimator = "grpo"
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.critic.model.path = None
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.resume_mode = None
    cfg.trainer.ckpt_interval = -1
    cfg.trainer.hf_save_interval = -1
    cfg.trainer.enable_ray_gpu_monitor = False
    cfg.trainer.logger = "console"
    cfg.generator.inference_engine.backend = "fireworks"
    cfg.generator.inference_engine.run_engines_locally = False
    cfg.generator.inference_engine.enable_ray_prometheus_stats = False
    cfg.generator.sampling_params.max_generate_length = 256
    cfg.generator.eval_sampling_params.max_generate_length = 256
    return cfg


def test_validate_fireworks_grpo_config() -> None:
    validate_cfg(_valid_fireworks_cfg())


def test_validate_fireworks_fully_async_grpo_config() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fully_async.enabled = True
    cfg.trainer.fully_async.max_staleness_steps = 0
    cfg.trainer.fully_async.num_parallel_generation_workers = 2
    cfg.trainer.train_batch_size = 2
    cfg.trainer.policy_mini_batch_size = 2
    cfg.generator.batched = False

    validate_cfg(cfg)


def test_validate_dedicated_fireworks_grpo_config() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.infrastructure = "dedicated"
    cfg.trainer.fireworks.training_shape_id = (
        "accounts/fireworks/trainingShapes/qwen3-4b-minimum-lora"
    )
    cfg.trainer.fireworks.trainer_job_id = "skyrl-smoke-test-trainer"
    cfg.trainer.fireworks.deployment_id = "skyrl-smoke-test-rollout"

    validate_cfg(cfg)


def test_validate_dedicated_full_parameter_fireworks_grpo_config() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.infrastructure = "dedicated"
    cfg.trainer.fireworks.training_shape_id = (
        "accounts/fireworks/trainingShapes/qwen3-4b-minimum"
    )
    cfg.trainer.fireworks.trainer_job_id = "skyrl-smoke-test-trainer"
    cfg.trainer.fireworks.deployment_id = "skyrl-smoke-test-rollout"
    cfg.trainer.fireworks.replica_count = 4
    cfg.trainer.policy.model.lora.rank = 0

    validate_cfg(cfg)


def test_validate_fireworks_checkpoint_and_resume_config() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.ckpt_interval = 1
    cfg.trainer.resume_mode = "from_path"
    cfg.trainer.resume_path = "/tmp/ckpts/global_step_7"

    validate_cfg(cfg)


def test_validate_fireworks_resume_from_path_requires_path() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.resume_mode = "from_path"

    with pytest.raises(ValueError, match="resume_path is required"):
        validate_cfg(cfg)


def test_validate_serverless_rejects_full_parameter_training() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.policy.model.lora.rank = 0

    with pytest.raises(ValueError, match="serverless training is LoRA-only"):
        validate_cfg(cfg)


def test_validate_dedicated_requires_positive_replica_count() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.infrastructure = "dedicated"
    cfg.trainer.fireworks.training_shape_id = (
        "accounts/fireworks/trainingShapes/qwen3-4b-minimum"
    )
    cfg.trainer.fireworks.trainer_job_id = "skyrl-smoke-test-trainer"
    cfg.trainer.fireworks.deployment_id = "skyrl-smoke-test-rollout"
    cfg.trainer.fireworks.replica_count = 0

    with pytest.raises(ValueError, match="replica_count > 0"):
        validate_cfg(cfg)


def test_validate_dedicated_requires_positive_trainer_replica_count() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.infrastructure = "dedicated"
    cfg.trainer.fireworks.training_shape_id = (
        "accounts/fireworks/trainingShapes/qwen3-4b-minimum"
    )
    cfg.trainer.fireworks.trainer_job_id = "skyrl-smoke-test-trainer"
    cfg.trainer.fireworks.deployment_id = "skyrl-smoke-test-rollout"
    cfg.trainer.fireworks.trainer_replica_count = 0

    with pytest.raises(ValueError, match="trainer_replica_count > 0"):
        validate_cfg(cfg)


def test_validate_dedicated_requires_auditable_resource_ids() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.infrastructure = "dedicated"

    with pytest.raises(ValueError, match="training_shape_id"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda cfg: setattr(cfg.trainer.algorithm, "advantage_estimator", "gae"),
            "GRPO-only",
        ),
        (
            lambda cfg: setattr(cfg.trainer.algorithm, "policy_loss_type", "regular"),
            "policy_loss_type='rollout_is'",
        ),
        (
            lambda cfg: setattr(cfg.trainer.critic.model, "path", "critic"),
            "policy-only",
        ),
        (
            lambda cfg: setattr(cfg.trainer.placement, "colocate_all", True),
            "cannot be colocated",
        ),
    ],
)
def test_validate_fireworks_rejects_out_of_scope_algorithms(
    mutate, message: str
) -> None:
    cfg = _valid_fireworks_cfg()
    mutate(cfg)

    with pytest.raises((ValueError, NotImplementedError), match=message):
        validate_cfg(cfg)


def test_fireworks_api_key_is_not_part_of_serialized_config(monkeypatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_secret_value")

    rendered = get_config_as_yaml_str(_valid_fireworks_cfg())

    assert "fw_secret_value" not in rendered
    assert "FIREWORKS_API_KEY" not in rendered
