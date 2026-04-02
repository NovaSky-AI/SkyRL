import pytest

from skyrl.tinker.skyrl_train_backend_config import (
    build_tinker_skyrl_train_config,
    collect_tinker_skyrl_train_backend_override_errors,
    validate_tinker_skyrl_train_backend_overrides,
)


BASE_MODEL = "trl-internal-testing/tiny-Qwen3ForCausalLM"


def _require_train_config_runtime() -> None:
    pytest.importorskip("omegaconf")
    pytest.importorskip("hydra")
    pytest.importorskip("skyrl_gym")


@pytest.mark.parametrize(
    ("key", "reason_snippet"),
    [
        ("trainer.policy.optimizer_config.lr", "OptimStepInput.adam_params.learning_rate"),
        ("trainer.algorithm.policy_loss_type", "ForwardBackwardInput.loss_fn"),
        ("trainer.policy.model.path", "--base-model"),
        ("trainer.policy.model.lora.rank", "CreateModelInput.lora_config.rank"),
        ("strategy", "--backend"),
        ("trainer.strategy", "--backend"),
    ],
)
def test_validate_tinker_skyrl_train_backend_overrides_rejects_high_confidence_keys(
    key: str, reason_snippet: str
):
    with pytest.raises(ValueError, match=key):
        validate_tinker_skyrl_train_backend_overrides("fsdp", {key: "value"})

    errors = dict(collect_tinker_skyrl_train_backend_override_errors("fsdp", {key: "value"}))
    assert reason_snippet in errors[key]


@pytest.mark.parametrize(
    ("backend", "key"),
    [
        ("fsdp", "trainer.policy.megatron_config.tensor_model_parallel_size"),
        ("megatron", "trainer.policy.fsdp_config.cpu_offload"),
    ],
)
def test_validate_tinker_skyrl_train_backend_overrides_rejects_backend_specific_namespaces(
    backend: str, key: str
):
    with pytest.raises(ValueError, match=key):
        validate_tinker_skyrl_train_backend_overrides(backend, {key: 1})


@pytest.mark.parametrize(
    "key",
    [
        "data.train_data",
        "trainer.ref.model.path",
        "trainer.critic.optimizer_config.lr",
        "trainer.fully_async.max_staleness_steps",
    ],
)
def test_validate_tinker_skyrl_train_backend_overrides_rejects_irrelevant_prefixes(key: str):
    with pytest.raises(ValueError, match=key):
        validate_tinker_skyrl_train_backend_overrides("fsdp", {key: "value"})


def test_build_tinker_skyrl_train_config_accepts_supported_startup_keys():
    _require_train_config_runtime()

    cfg = build_tinker_skyrl_train_config(
        base_model=BASE_MODEL,
        backend="fsdp",
        overrides={
            "trainer.algorithm.temperature": 0.7,
            "trainer.policy.optimizer_config.weight_decay": 0.123,
            "trainer.placement.policy_num_gpus_per_node": 2,
            "generator.inference_engine.num_engines": 2,
        },
    )

    assert cfg.trainer.algorithm.temperature == 0.7
    assert cfg.trainer.policy.optimizer_config.weight_decay == 0.123
    assert cfg.trainer.placement.policy_num_gpus_per_node == 2
    assert cfg.generator.inference_engine.num_engines == 2
    assert cfg.trainer.policy.model.path == BASE_MODEL
    assert cfg.trainer.policy.optimizer_config.scheduler == "constant_with_warmup"
    assert cfg.trainer.policy.optimizer_config.num_warmup_steps == 0
    assert cfg.trainer.algorithm.use_kl_loss is False
    assert cfg.trainer.strategy == "fsdp2"


def test_build_tinker_skyrl_train_config_still_uses_typed_schema_validation():
    _require_train_config_runtime()

    with pytest.raises(ValueError, match="Invalid fields"):
        build_tinker_skyrl_train_config(
            base_model=BASE_MODEL,
            backend="fsdp",
            overrides={"trainer.policy.not_a_real_field": 1},
        )
