import pytest

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import (
    prepare_runtime_environment,
    validate_logprob_comparison,
)
from tests.train.util import example_dummy_config


def _full_mode_config():
    cfg = example_dummy_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.enable_isoexec = True
    cfg.trainer.rollout_logprob_comparison = "full"
    cfg.trainer.remove_microbatch_padding = True
    cfg.generator.inference_engine.logprob_output = "full"
    cfg.generator.batched = True
    return cfg


def test_isoexec_is_opt_in_and_preserves_default_runtime_environment():
    cfg = example_dummy_config()

    assert cfg.trainer.enable_isoexec is False
    assert "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES" not in prepare_runtime_environment(cfg)


def test_isoexec_runtime_uses_physical_gpu_namespace():
    cfg = example_dummy_config()
    cfg.trainer.enable_isoexec = True

    assert prepare_runtime_environment(cfg)["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] == "1"


def test_isoexec_runtime_forwards_isoexec_overrides_only_when_enabled(monkeypatch):
    monkeypatch.setenv("ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR", "/evidence/rows")
    monkeypatch.setenv("ISOEXEC", "1")
    cfg = example_dummy_config()

    assert "ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR" not in prepare_runtime_environment(cfg)

    cfg.trainer.enable_isoexec = True
    env_vars = prepare_runtime_environment(cfg)

    assert env_vars["ISOEXEC_FULL_DISTRIBUTION_EVIDENCE_DIR"] == "/evidence/rows"
    assert env_vars["ISOEXEC"] == "1"


def test_full_mode_fields_are_cli_overridable():
    cfg = SkyRLTrainConfig.from_cli_overrides(
        [
            "trainer.enable_isoexec=true",
            "trainer.rollout_logprob_comparison=full",
            "trainer.remove_microbatch_padding=true",
            "generator.inference_engine.logprob_output=full",
        ]
    )

    assert cfg.trainer.enable_isoexec is True
    assert cfg.trainer.rollout_logprob_comparison == "full"
    assert cfg.trainer.remove_microbatch_padding is True
    assert cfg.generator.inference_engine.logprob_output == "full"


def test_action_mode_preserves_evaluation_logprobs():
    cfg = example_dummy_config()
    assert cfg.generator.eval_sampling_params.logprobs == 1

    validate_logprob_comparison(cfg)

    assert cfg.generator.eval_sampling_params.logprobs == 1


def test_full_mode_accepts_narrow_megatron_vllm_profile():
    cfg = _full_mode_config()

    validate_logprob_comparison(cfg)

    assert cfg.generator.eval_sampling_params.logprobs is None


def test_logprob_modes_must_match():
    cfg = example_dummy_config()
    cfg.trainer.rollout_logprob_comparison = "full"

    with pytest.raises(ValueError, match="must agree"):
        validate_logprob_comparison(cfg)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda cfg: setattr(cfg.trainer, "strategy", "fsdp"),
        lambda cfg: setattr(cfg.generator, "batched", False),
        lambda cfg: setattr(cfg.trainer.fully_async, "enabled", True),
        lambda cfg: setattr(cfg.trainer.mtp, "enabled", True),
        lambda cfg: setattr(cfg.trainer, "fused_lm_head_logprob", True),
        lambda cfg: setattr(cfg.trainer, "enable_isoexec", False),
        lambda cfg: setattr(cfg.trainer, "remove_microbatch_padding", False),
        # Non-colocated is supported only with NCCL broadcast weight sync.
        lambda cfg: (
            setattr(cfg.trainer.placement, "colocate_all", False),
            setattr(cfg.generator.inference_engine, "weight_sync_backend", "delta"),
        ),
        lambda cfg: setattr(cfg.generator, "vision_language_generator", True),
        lambda cfg: setattr(cfg.trainer.policy.megatron_config, "expert_tensor_parallel_size", 2),
        # Engine tensor parallelism is admitted only non-colocated (see the dedicated test below).
        lambda cfg: setattr(cfg.generator.inference_engine, "pipeline_parallel_size", 2),
        lambda cfg: setattr(cfg.generator.inference_engine, "expert_parallel_size", 2),
        lambda cfg: setattr(cfg.generator.sampling_params, "temperature", 0.5),
        lambda cfg: setattr(cfg.generator.sampling_params, "logprobs", None),
    ],
)
def test_full_mode_rejects_unsupported_profiles(mutate):
    cfg = _full_mode_config()
    mutate(cfg)

    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_data_parallel_trainer_only_with_asymmetric_colocation():
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = 2

    with pytest.raises(ValueError, match="asymmetric_colocation"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.asymmetric_colocation = True
    validate_logprob_comparison(cfg)

    # Tensor parallelism over the two policy GPUs is admitted under the same contract.
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    validate_logprob_comparison(cfg)

    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 4
    with pytest.raises(ValueError, match="one GPU per side"):
        validate_logprob_comparison(cfg)

    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.placement.asymmetric_colocation = False
    with pytest.raises(ValueError, match="asymmetric_colocation"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_non_colocated_data_parallel_engines_with_nccl_broadcast():
    cfg = _full_mode_config()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    cfg.generator.inference_engine.data_parallel_size = 2
    validate_logprob_comparison(cfg)

    cfg.generator.inference_engine.weight_sync_backend = "delta"
    with pytest.raises(ValueError, match="NCCL broadcast"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_independent_engine_replicas_only_when_non_colocated():
    cfg = _full_mode_config()
    cfg.generator.inference_engine.num_engines = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    validate_logprob_comparison(cfg)


def test_full_mode_admits_expert_parallel_trainer_only_with_asymmetric_colocation():
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.asymmetric_colocation = True
    validate_logprob_comparison(cfg)

    # EP must span exactly the policy GPUs, with TP=1 (IsoExec: EP = TP x dense DP, expert TP 1).
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 4
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_tensor_parallel_engine_only_when_non_colocated():
    cfg = _full_mode_config()
    cfg.generator.inference_engine.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    validate_logprob_comparison(cfg)

    # Pipeline parallelism on the engine stays refused in either placement.
    cfg.generator.inference_engine.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)


def test_full_mode_rejects_data_parallel_engines_when_colocated():
    cfg = _full_mode_config()
    cfg.generator.inference_engine.data_parallel_size = 2
    with pytest.raises(ValueError, match="tensor-parallel or replicated engines"):
        validate_logprob_comparison(cfg)
