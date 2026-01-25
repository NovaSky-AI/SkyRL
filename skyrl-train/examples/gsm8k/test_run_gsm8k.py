#!/usr/bin/env python3
"""
Quick test run of GSM8K with Pydantic config to validate the new configuration system.

This is a minimal test run (1 epoch, console logging) to verify the config works correctly.

Usage:
    CUDA_VISIBLE_DEVICES=4,5,6,7 uv run --extra vllm examples/gsm8k/test_run_gsm8k.py
"""

import os
from pathlib import Path

from skyrl_train.config.configs import create_default_config
from skyrl_train.entrypoints.main_base import run_training


def get_test_config():
    """Create a minimal GSM8K test configuration."""
    from skyrl_train.config.configs import (
        SkyRLConfig, DataConfig, TrainerConfig, GeneratorConfig, EnvironmentConfig,
        PlacementConfig, PolicyConfig, RefConfig, CriticConfig,
        ModelConfig, CriticModelConfig, OptimizerConfig, AlgorithmConfig,
        SamplingParamsConfig
    )

    data_dir = os.environ.get("DATA_DIR", f"{Path.home()}/data/gsm8k")
    model_path = "Qwen/Qwen2.5-1.5B-Instruct"

    # Build configuration compositionally - minimal settings for quick test
    return SkyRLConfig(
        data=DataConfig(
            train_data=[f"{data_dir}/train.parquet"],
            val_data=[f"{data_dir}/validation.parquet"],
        ),
        trainer=TrainerConfig(
            placement=PlacementConfig(
                colocate_all=True,
                policy_num_gpus_per_node=4,
                critic_num_gpus_per_node=4,
                ref_num_gpus_per_node=4,
            ),
            strategy="fsdp2",
            policy=PolicyConfig(
                model=ModelConfig(path=model_path),
                optimizer_config=OptimizerConfig(lr=1.0e-6),
            ),
            ref=RefConfig(
                model=ModelConfig(path=model_path),
            ),
            critic=CriticConfig(
                model=CriticModelConfig(path=model_path),
                optimizer_config=OptimizerConfig(lr=5.0e-6),
            ),
            algorithm=AlgorithmConfig(
                advantage_estimator="grpo",
                use_kl_loss=True,
            ),
            epochs=1,  # Just 1 epoch for testing
            train_batch_size=256,  # Smaller batch for faster test
            policy_mini_batch_size=128,
            micro_forward_batch_size_per_gpu=32,
            micro_train_batch_size_per_gpu=32,
            ckpt_interval=100,  # Don't checkpoint during short test
            ckpt_path=f"{Path.home()}/ckpts/test_pydantic_config",
            export_path=f"{Path.home()}/exports/",
            eval_batch_size=256,
            eval_before_train=False,  # Skip initial eval for speed
            eval_interval=-1,  # Disable periodic eval
            max_prompt_length=512,
            logger="console",
            project_name="gsm8k_test",
            run_name="pydantic_config_test",
            resume_mode=None,
        ),
        generator=GeneratorConfig(
            model_name=model_path,
            num_inference_engines=4,
            inference_engine_tensor_parallel_size=1,
            backend="vllm",
            run_engines_locally=True,
            weight_sync_backend="nccl",
            async_engine=True,
            batched=True,
            n_samples_per_prompt=5,
            gpu_memory_utilization=0.8,
            max_input_length=512,
            sampling_params=SamplingParamsConfig(
                max_generate_length=1024,
            ),
            eval_sampling_params=SamplingParamsConfig(
                max_generate_length=1024,
                temperature=0.0,
            ),
        ),
        environment=EnvironmentConfig(
            env_class="gsm8k",
        ),
    )


def main():
    """Main entrypoint for test run."""
    print("=" * 80)
    print("TESTING PYDANTIC CONFIG SYSTEM")
    print("=" * 80)
    print("This is a minimal test run to validate the new configuration system.")
    print("Running 1 epoch with small batch sizes and console logging.")
    print("=" * 80)

    cfg = get_test_config()

    # Print key config values
    print("\nConfiguration Summary:")
    print(f"  Model: {cfg.trainer.policy.model.path}")
    print(f"  Algorithm: {cfg.trainer.algorithm.advantage_estimator}")
    print(f"  Epochs: {cfg.trainer.epochs}")
    print(f"  Train Batch Size: {cfg.trainer.train_batch_size}")
    print(f"  GPUs: {cfg.trainer.placement.policy_num_gpus_per_node}")
    print(f"  Backend: {cfg.generator.backend}")
    print(f"  Logger: {cfg.trainer.logger}")
    print("=" * 80 + "\n")

    # Run training
    run_training(cfg)


if __name__ == "__main__":
    main()
