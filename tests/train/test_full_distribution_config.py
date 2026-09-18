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
        # A colocated engine cannot outgrow the policy GPUs, whichever degree multiplies it (pipeline
        # stages included); a degree that is not a positive integer refuses on its own.
        lambda cfg: setattr(cfg.generator.inference_engine, "pipeline_parallel_size", 2),
        lambda cfg: setattr(cfg.generator.inference_engine, "pipeline_parallel_size", 0),
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
    with pytest.raises(ValueError, match="dividing the 2 policy GPUs"):
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
    with pytest.raises(ValueError, match="weight_sync_backend=nccl when not colocated"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_engine_expert_parallelism_only_across_non_colocated_data_parallel_ranks():
    cfg = _full_mode_config()
    engine = cfg.generator.inference_engine
    cfg.trainer.placement.colocate_all = False
    engine.weight_sync_backend = "nccl"
    engine.data_parallel_size = 2
    engine.expert_parallel_size = 2  # = data_parallel_size x tensor_parallel_size: IsoExec's own dispatch
    validate_logprob_comparison(cfg)

    engine.expert_parallel_size = 4  # not DP x TP
    with pytest.raises(ValueError, match="expert_parallel_size=1, or = data_parallel_size x tensor_parallel_size"):
        validate_logprob_comparison(cfg)
    engine.expert_parallel_size = 2
    engine.data_parallel_size = 1  # EP without the DP ranks it spans
    with pytest.raises(ValueError, match="expert_parallel_size=1, or = data_parallel_size x tensor_parallel_size"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_independent_engine_replicas_only_when_non_colocated():
    cfg = _full_mode_config()
    cfg.generator.inference_engine.num_engines = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    validate_logprob_comparison(cfg)


def test_full_mode_admits_pipeline_parallel_trainer_only_with_asymmetric_colocation():
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.asymmetric_colocation = True
    validate_logprob_comparison(cfg)

    # PP spans exactly the policy GPUs, with TP=1 and EP=1.
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 4
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.context_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_context_parallel_at_degree_two_only():
    # CP=2 on two policy GPUs: one replica's tokens are cut over the pair, the engine keeps GPU 0.
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.placement.asymmetric_colocation = True
    cfg.trainer.policy.megatron_config.context_parallel_size = 2
    validate_logprob_comparison(cfg)

    # It composes with TP and PP as one more model-parallel factor of the policy GPUs.
    cfg.trainer.placement.policy_num_gpus_per_node = 8
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    validate_logprob_comparison(cfg)

    # Any other degree is refused by name, and so is expert parallelism under a CP cut.
    cfg.trainer.policy.megatron_config.context_parallel_size = 4
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
    with pytest.raises(ValueError, match="context_parallel_size in \\(1, 2\\)"):
        validate_logprob_comparison(cfg)
    cfg.trainer.policy.megatron_config.context_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 2
    with pytest.raises(ValueError, match="expert_model_parallel_size=1 under context parallelism"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_expert_parallel_trainer_only_with_asymmetric_colocation():
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.asymmetric_colocation = True
    validate_logprob_comparison(cfg)

    # EP spans exactly the policy GPUs (IsoExec: EP = TP x dense DP, expert TP 1); TP may be 1
    # (dense-DP replicas) or the policy GPUs (one TP group whose ranks are also the expert owners).
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    validate_logprob_comparison(cfg)
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 4
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)
    cfg.trainer.policy.megatron_config.expert_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_trainer_compositions_that_tile_the_policy_gpus():
    cfg = _full_mode_config()
    mc = cfg.trainer.policy.megatron_config
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.placement.asymmetric_colocation = True
    mc.tensor_model_parallel_size, mc.pipeline_model_parallel_size = 2, 2  # TP x PP, dense DP 1
    validate_logprob_comparison(cfg)
    mc.transformer_config_kwargs["sequence_parallel"] = True  # + SP on the TP axis
    validate_logprob_comparison(cfg)
    mc.pipeline_model_parallel_size = 1  # TP x DP (+SP), and EP = TP x dense DP = 4
    mc.expert_model_parallel_size = 4
    validate_logprob_comparison(cfg)
    mc.expert_model_parallel_size = 2  # EP = TP: two replicas of each expert shard (expert DP 2)
    validate_logprob_comparison(cfg)
    mc.expert_model_parallel_size = 3  # neither a multiple of TP nor a divisor of TP x dense DP
    with pytest.raises(ValueError, match="expert_model_parallel_size 1, or a multiple of TP"):
        validate_logprob_comparison(cfg)
    mc.expert_model_parallel_size = 1
    mc.tensor_model_parallel_size = 1  # SP without TP
    with pytest.raises(ValueError, match="sequence_parallel"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_colocated_engines_on_a_prefix_of_the_policy_gpus():
    cfg = _full_mode_config()
    engine = cfg.generator.inference_engine
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    engine.tensor_parallel_size = 2  # one TP=2 engine on policy GPUs 0-1
    with pytest.raises(ValueError, match="asymmetric_colocation=true"):
        validate_logprob_comparison(cfg)
    cfg.trainer.placement.asymmetric_colocation = True
    validate_logprob_comparison(cfg)
    engine.num_engines = 2  # two TP=2 replicas tile all four GPUs: no asymmetry needed
    cfg.trainer.placement.asymmetric_colocation = False
    validate_logprob_comparison(cfg)
    engine.num_engines = 3
    with pytest.raises(ValueError, match="fitting the 4 policy GPUs"):
        validate_logprob_comparison(cfg)


def test_full_mode_admits_tensor_parallel_engine_only_when_non_colocated():
    cfg = _full_mode_config()
    cfg.generator.inference_engine.tensor_parallel_size = 2
    with pytest.raises(ValueError, match="full logprob comparison requires"):
        validate_logprob_comparison(cfg)

    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine.weight_sync_backend = "nccl"
    validate_logprob_comparison(cfg)

    # Engine pipeline stages compose with it: every worker of every stage receives the weight stream.
    cfg.generator.inference_engine.pipeline_parallel_size = 2
    validate_logprob_comparison(cfg)


def test_full_mode_admits_engine_pipeline_stages_inside_the_policy_gpus():
    # Colocated: TP2 x PP2 engine on the four GPUs of a TP2 x PP2 trainer.
    cfg = _full_mode_config()
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
    cfg.generator.inference_engine.tensor_parallel_size = 2
    cfg.generator.inference_engine.pipeline_parallel_size = 2
    validate_logprob_comparison(cfg)

    # One more stage than GPUs is refused by the colocation rule, not by a pipeline rule.
    cfg.generator.inference_engine.pipeline_parallel_size = 4
    with pytest.raises(ValueError, match="colocated engines \\(8 GPUs\\) fitting the 4 policy GPUs"):
        validate_logprob_comparison(cfg)

    cfg.generator.inference_engine.pipeline_parallel_size = 0
    with pytest.raises(ValueError, match="pipeline_parallel_size >= 1"):
        validate_logprob_comparison(cfg)


def test_full_mode_rejects_data_parallel_engines_when_colocated():
    cfg = _full_mode_config()
    cfg.generator.inference_engine.data_parallel_size = 2
    with pytest.raises(ValueError, match="data_parallel_size=1 when colocated"):
        validate_logprob_comparison(cfg)
