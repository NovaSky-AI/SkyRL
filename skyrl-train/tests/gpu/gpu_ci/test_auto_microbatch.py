"""
Tests for automatic micro-batch size determination.

Run with:
    uv run --isolated --extra dev -- pytest tests/gpu/gpu_ci/test_auto_microbatch.py -v

Requires: 2 GPUs minimum, 4 GPUs for HYBRID_SHARD test.
"""

import ray
import pytest
import torch

from tests.gpu.utils import init_worker_with_type
from skyrl_train.config import SkyRLConfig


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def get_auto_test_config(
    num_gpus: int = 2,
    fsdp_size: int = -1,
    strategy: str = "fsdp2",
) -> SkyRLConfig:
    """Build a SkyRLConfig with ``auto_micro_batch_size`` enabled."""
    cfg = SkyRLConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = num_gpus
    cfg.generator.inference_engine_tensor_parallel_size = num_gpus
    cfg.trainer.strategy = strategy
    cfg.trainer.policy.fsdp_config.fsdp_size = fsdp_size
    cfg.trainer.logger = "console"
    cfg.trainer.auto_micro_batch_size = True
    cfg.trainer.policy_mini_batch_size = 16
    cfg.trainer.train_batch_size = 16
    cfg.generator.n_samples_per_prompt = 1
    return cfg


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["fsdp2"], ids=["fsdp2"])
async def test_auto_microbatch_full_shard(ray_init_fixture, strategy):
    """2 GPUs, model fully sharded — all workers must agree on the batch size."""
    cfg = get_auto_test_config(num_gpus=2, fsdp_size=-1, strategy=strategy)

    try:
        actor_group = init_worker_with_type(
            "policy",
            num_gpus_per_node=2,
            cfg=cfg,
        )

        # Compute the same mini_batch_size_per_gpu the trainer would use
        dp_size = cfg.trainer.placement.policy_num_gpus_per_node // cfg.trainer.policy.sequence_parallel_size
        mini_batch_size_per_gpu = (
            cfg.trainer.policy_mini_batch_size * cfg.generator.n_samples_per_prompt // dp_size
        )
        max_seq_len = cfg.trainer.max_prompt_length + cfg.generator.sampling_params.max_generate_length

        # Run auto sizing on all workers
        results = ray.get(
            actor_group.async_run_ray_method(
                "pass_through",
                "auto_determine_micro_batch_size",
                max_seq_len,
                mini_batch_size_per_gpu,
            )
        )

        # All workers should return the same value (coordinated via all_reduce)
        assert all(r == results[0] for r in results), f"Workers disagree: {results}"
        assert results[0] >= 1, "Micro batch size must be at least 1"

        # Must divide mini_batch_size_per_gpu evenly
        assert mini_batch_size_per_gpu % results[0] == 0, (
            f"Determined size {results[0]} doesn't divide mini_batch_size_per_gpu {mini_batch_size_per_gpu}"
        )

        print(f"[{strategy}] Determined micro batch size: {results[0]}")
    finally:
        ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="Need 4 GPUs for HYBRID_SHARD test",
)
async def test_auto_microbatch_hybrid_shard(ray_init_fixture):
    """4 GPUs, fsdp_size=2 — HYBRID_SHARD with 2 FSDP groups of 2."""
    cfg = get_auto_test_config(num_gpus=4, fsdp_size=2, strategy="fsdp2")

    try:
        actor_group = init_worker_with_type(
            "policy",
            num_gpus_per_node=4,
            cfg=cfg,
        )

        dp_size = cfg.trainer.placement.policy_num_gpus_per_node // cfg.trainer.policy.sequence_parallel_size
        mini_batch_size_per_gpu = (
            cfg.trainer.policy_mini_batch_size * cfg.generator.n_samples_per_prompt // dp_size
        )
        max_seq_len = cfg.trainer.max_prompt_length + cfg.generator.sampling_params.max_generate_length

        results = ray.get(
            actor_group.async_run_ray_method(
                "pass_through",
                "auto_determine_micro_batch_size",
                max_seq_len,
                mini_batch_size_per_gpu,
            )
        )

        assert all(r == results[0] for r in results), f"Workers disagree: {results}"
        assert results[0] >= 1
        assert mini_batch_size_per_gpu % results[0] == 0

        print(f"[hybrid] Determined micro batch size: {results[0]}")
    finally:
        ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["fsdp2"], ids=["fsdp2"])
async def test_auto_microbatch_deterministic(ray_init_fixture, strategy):
    """Run profiling twice — both runs should produce the same result."""
    cfg = get_auto_test_config(num_gpus=2, fsdp_size=-1, strategy=strategy)

    try:
        actor_group = init_worker_with_type(
            "policy",
            num_gpus_per_node=2,
            cfg=cfg,
        )

        dp_size = cfg.trainer.placement.policy_num_gpus_per_node // cfg.trainer.policy.sequence_parallel_size
        mini_batch_size_per_gpu = (
            cfg.trainer.policy_mini_batch_size * cfg.generator.n_samples_per_prompt // dp_size
        )
        max_seq_len = cfg.trainer.max_prompt_length + cfg.generator.sampling_params.max_generate_length

        # First run
        results_1 = ray.get(
            actor_group.async_run_ray_method(
                "pass_through",
                "auto_determine_micro_batch_size",
                max_seq_len,
                mini_batch_size_per_gpu,
            )
        )

        # Second run
        results_2 = ray.get(
            actor_group.async_run_ray_method(
                "pass_through",
                "auto_determine_micro_batch_size",
                max_seq_len,
                mini_batch_size_per_gpu,
            )
        )

        assert results_1 == results_2, (
            f"Non-deterministic results: run1={results_1}, run2={results_2}"
        )
    finally:
        ray.shutdown()
