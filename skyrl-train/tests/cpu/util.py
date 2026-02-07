# utility functions used for CPU tests

from skyrl_train.config import SkyRLConfig, AlgorithmConfig, SamplingParams, TrainerConfig, GeneratorConfig


def example_dummy_config():
    # TODO (sumanthrh): Cleanup overrides
    trainer_cfg = TrainerConfig(
        project_name="unit-test",
        run_name="test-run",
        logger="tensorboard",
        micro_train_batch_size_per_gpu=2,
        train_batch_size=2,
        eval_batch_size=2,
        update_epochs_per_batch=1,
        epochs=1,
        max_prompt_length=20,
        use_sample_packing=False,
        seed=42,
        resume_mode="none",
        algorithm=AlgorithmConfig(
            advantage_estimator="grpo",
            kl_estimator_type="k1",
            use_kl_loss=True,
            kl_loss_coef=0.0,
            loss_reduction="token_mean",
            grpo_norm_by_std=True,
        ),
    )
    generator_cfg = GeneratorConfig(
        sampling_params=SamplingParams(max_generate_length=20),
        n_samples_per_prompt=1,
        batched=False,
        max_turns=1,
        enable_http_endpoint=False,
        http_endpoint_host="127.0.0.1",
        http_endpoint_port=8000,
    )
    cfg = SkyRLConfig(trainer=trainer_cfg, generator=generator_cfg)

    return cfg
