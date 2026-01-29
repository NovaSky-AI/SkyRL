#!/usr/bin/env python3
"""
Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on GSM8K.

This is a Python-based configuration example that uses Pydantic models
instead of YAML+Hydra CLI overrides.

Setup:
    uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
    export WANDB_API_KEY=<your_key_here>

Usage:
    # With default settings (4 GPUs, vLLM, wandb logging):
    uv run --extra vllm examples/gsm8k/run_gsm8k.py

    # With CLI overrides (any config field via --key.path=value):
    uv run --extra vllm examples/gsm8k/run_gsm8k.py \
        --trainer.epochs=30 \
        --trainer.policy.model.path="Qwen/Qwen2.5-7B" \
        --trainer.algorithm.advantage_estimator="gae"

    # With environment variables for convenience params:
    NUM_GPUS=8 LOGGER=console INFERENCE_BACKEND=sglang \
        uv run --extra sglang examples/gsm8k/run_gsm8k.py

    # Programmatic usage:
    from examples.gsm8k.run_gsm8k import get_gsm8k_config
    cfg = get_gsm8k_config(num_gpus=8)
    # cfg is a fully typed SkyRLConfig object
"""

import os
from pathlib import Path

from skyrl_train.config.configs import SkyRLConfig
from skyrl_train.entrypoints.main_base import run_training


def get_gsm8k_config(
    data_dir: str = None,
    num_gpus: int = 4,
    logger: str = "wandb",
    inference_backend: str = "vllm",
) -> SkyRLConfig:
    """
    Create GSM8K training configuration.

    Args:
        data_dir: Path to GSM8K data directory (default: $HOME/data/gsm8k)
        num_gpus: Number of GPUs to use (default: 4)
        logger: Logging backend - "wandb" or "console" (default: "wandb")
        inference_backend: Inference backend - "vllm" or "sglang" (default: "vllm")

    Returns:
        SkyRLConfig: Configured training configuration
    """
    from skyrl_train.config.configs import (
        DataConfig, TrainerConfig, GeneratorConfig, EnvironmentConfig,
        PlacementConfig, PolicyConfig, RefConfig, CriticConfig,
        ModelConfig, CriticModelConfig, OptimizerConfig, AlgorithmConfig,
        SamplingParamsConfig
    )

    # Get data directory from environment or use default
    if data_dir is None:
        data_dir = os.environ.get("DATA_DIR", f"{Path.home()}/data/gsm8k")

    model_path = "Qwen/Qwen2.5-1.5B-Instruct"

    # ========================================================================
    # Build configuration compositionally
    # ========================================================================

    return SkyRLConfig(
        data=DataConfig(
            train_data=[f"{data_dir}/train.parquet"],
            val_data=[f"{data_dir}/validation.parquet"],
        ),
        trainer=TrainerConfig(
            placement=PlacementConfig(
                colocate_all=True,
                policy_num_gpus_per_node=num_gpus,
                critic_num_gpus_per_node=num_gpus,
                ref_num_gpus_per_node=num_gpus,
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
            epochs=20,
            train_batch_size=1024,
            policy_mini_batch_size=256,
            micro_forward_batch_size_per_gpu=64,
            micro_train_batch_size_per_gpu=64,
            ckpt_interval=10,
            ckpt_path=f"{Path.home()}/ckpts/gsm8k_1.5B_ckpt",
            export_path=f"{Path.home()}/exports/",
            eval_batch_size=1024,
            eval_before_train=True,
            eval_interval=5,
            max_prompt_length=512,
            logger=logger,
            project_name="gsm8k",
            run_name="gsm8k_test",
            resume_mode=None,
        ),
        generator=GeneratorConfig(
            model_name=model_path,
            num_inference_engines=num_gpus,
            inference_engine_tensor_parallel_size=1,
            backend=inference_backend,
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
    """Main entrypoint for GSM8K training with Pydantic config."""
    import sys
    from skyrl_train.config.cli import apply_overrides

    # Collect all --key=value args as config overrides
    overrides = []
    for arg in sys.argv[1:]:
        if arg.startswith("--") and "=" in arg:
            overrides.append(arg[2:])  # Strip leading --

    # Read base settings from environment variables
    # Note: num_gpus is a convenience param that sets multiple config fields
    # (placement and generator). Use env vars or override each field directly.
    data_dir = os.environ.get("DATA_DIR", None)
    num_gpus = int(os.environ.get("NUM_GPUS", "4"))
    logger = os.environ.get("LOGGER", "wandb")
    inference_backend = os.environ.get("INFERENCE_BACKEND", "vllm")

    # Create base configuration
    cfg = get_gsm8k_config(
        data_dir=data_dir,
        num_gpus=num_gpus,
        logger=logger,
        inference_backend=inference_backend,
    )

    # Apply CLI overrides (e.g., --trainer.epochs=30)
    if overrides:
        print(f"Applying {len(overrides)} config override(s):")
        for override in overrides:
            print(f"  {override}")
        cfg = apply_overrides(cfg, overrides)

    # Run training
    run_training(cfg)


if __name__ == "__main__":
    main()
