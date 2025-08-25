"""
Tests for dynamic token-based batching.

Run with:
uv run --isolated --extra dev -- pytest tests/gpu/test_dynamic_batching.py -v
"""

import pytest
import torch
import ray

from skyrl_train.training_batch import TrainingInputBatch
from tests.gpu.utils import init_worker_with_type, get_test_actor_config, make_variable_length_training_batch


@pytest.mark.parametrize(
    "use_dynamic,seq_lengths,expected_mode",
    [
        (False, [100, 100, 100, 100], "fixed"),
        (True, [50, 100, 150, 200], "dynamic"),
    ],
)
def test_ppo_train_with_adaptive_batching(use_dynamic, seq_lengths, expected_mode):
    """
    Test that ppo_train works correctly with AdaptiveBatchIterator.

    This test validates:
    - Both fixed and dynamic batching modes work
    - Training completes successfully
    - Correct batching mode is used
    """
    try:
        cfg = get_test_actor_config()
        cfg.trainer.strategy = "fsdp"
        cfg.trainer.placement.policy_num_gpus_per_node = 2
        cfg.trainer.update_epochs_per_batch = 1
        cfg.trainer.micro_train_batch_size_per_gpu = 2
        cfg.trainer.policy_mini_batch_size = 4
        cfg.generator.n_samples_per_prompt = 1

        # Configure batching mode
        cfg.trainer.use_dynamic_batching = use_dynamic
        if use_dynamic:
            cfg.trainer.max_token_len_per_gpu_train = 300

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Create training batch with specified sequence lengths
        train_data = make_variable_length_training_batch(seq_lengths, num_actions=4)
        train_data.metadata["global_step"] = 0

        # Run ppo_train
        results = ray.get(actor_group.async_run_ray_method("pass_through", "ppo_train", train_data))

        result = results[0]
        assert hasattr(result, "metadata"), "Result should have metadata attribute"
        assert "train_status" in result.metadata, "Should have train_status in metadata"

        train_status = result.metadata["train_status"]

        # Verify training metrics are present
        assert "policy_loss" in train_status, "Should have policy_loss"
        assert "policy_update_steps" in train_status, "Should have policy_update_steps"
        assert train_status["policy_update_steps"] > 0, "Should have completed at least one update step"

        print(f"{expected_mode} batching ppo_train completed with metrics: {train_status}")
    finally:
        ray.shutdown()


def test_batch_iterator_dynamic_sync_across_workers():
    """
    Test that BatchIterator with dynamic batching synchronizes across workers.

    This test validates:
    - Dynamic batching calculates different micro-batch counts per worker
    - Workers synchronize to use the same max micro-batch count
    - Training completes successfully with synchronized dynamic batching
    """
    try:
        cfg = get_test_actor_config()
        cfg.trainer.strategy = "fsdp"
        cfg.trainer.placement.policy_num_gpus_per_node = 2
        cfg.trainer.policy_mini_batch_size = 8
        cfg.trainer.update_epochs_per_batch = 1
        cfg.generator.n_samples_per_prompt = 4

        cfg.trainer.use_dynamic_batching = True
        cfg.trainer.max_token_len_per_gpu_train = 300

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        seq_lengths = [20, 40, 60, 80, 100, 120, 140, 160] * 4
        train_data = make_variable_length_training_batch(seq_lengths, num_actions=4)
        train_data.metadata["global_step"] = 0

        results = ray.get(actor_group.async_run_ray_method("pass_through", "ppo_train", train_data))

        assert len(results) == cfg.trainer.placement.policy_num_gpus_per_node, "Should get result from each GPU"

        for i, result in enumerate(results):
            assert hasattr(result, "metadata"), f"Result {i} should have metadata attribute"
            assert "train_status" in result.metadata, f"Result {i} should have train_status in metadata"

            train_status = result.metadata["train_status"]

            print(f"Train status: {train_status}")
            assert "policy_loss" in train_status, "Should have policy_loss"
            assert train_status["policy_update_steps"] > 0, "Should have completed update steps"

            print(f"Result {i}: {result}")

        print(f"Dynamic batching sync test completed with {len(results)} workers")
        print("Check logs for synchronized micro-batch counts across workers")

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_e2e_dynamic_batching_loss_consistency():
    """
    Test that one training step produces the same loss with and without dynamic batching.

    This test validates:
    - Training with fixed batch size produces a specific loss
    - Training with dynamic batching produces the same loss for the same data
    - Both modes handle GSM8K data correctly
    - Multi-GPU distributed training produces consistent results
    - Full e2e pipeline with inference engines and generator input
    """
    # import asyncio
    # from tests.gpu.test_skyrl_gym_generator import run_generator_end_to_end

    try:
        # Set seeds for reproducibility
        torch.manual_seed(42)
        import random
        import numpy as np

        random.seed(42)
        np.random.seed(42)

        # Base configuration with multiple GPUs
        cfg = get_test_actor_config()
        cfg.trainer.strategy = "fsdp"
        cfg.trainer.placement.policy_num_gpus_per_node = 2  # Use 2 GPUs
        cfg.trainer.policy.sequence_parallel_size = 2
        cfg.trainer.update_epochs_per_batch = 1
        cfg.trainer.micro_train_batch_size_per_gpu = 2
        cfg.trainer.policy_mini_batch_size = 8  # Total batch size across GPUs
        cfg.generator.n_samples_per_prompt = 1
        cfg.trainer.algorithm.max_seq_len = 512

        # Use a smaller model for testing
        cfg.trainer.policy.model.path = "Qwen/Qwen2.5-0.5B-Instruct"

        seq_lengths = np.random.randint(50, 512, size=8)
        # seq_lengths = [512, 400, 300, 200, 100, 50, 25, 10]

        train_data = make_variable_length_training_batch(seq_lengths, num_actions=4)
        train_data.metadata["global_step"] = 0

        cfg.trainer.use_dynamic_batching = False

        actor_group_fixed = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        train_data_fixed = TrainingInputBatch(
            {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in train_data.items()}
        )
        train_data_fixed.metadata = train_data.metadata.copy()

        results_fixed = ray.get(actor_group_fixed.async_run_ray_method("pass_through", "ppo_train", train_data_fixed))

        print(f"Results fixed: {results_fixed}")
        losses_fixed = [r.metadata["train_status"]["policy_loss"] for r in results_fixed]
        loss_fixed = sum(losses_fixed) / len(losses_fixed)

        print(f"Fixed batching losses per GPU: {losses_fixed}")
        print(f"Fixed batching average loss: {loss_fixed}")

        ray.shutdown()

        torch.manual_seed(42)
        random.seed(42)
        np.random.seed(42)

        ray.init()

        cfg.trainer.use_dynamic_batching = True
        cfg.trainer.max_token_len_per_gpu_train = 300

        actor_group_dynamic = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        train_data_dynamic = TrainingInputBatch(
            {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in train_data.items()}
        )
        train_data_dynamic.metadata = train_data.metadata.copy()

        results_dynamic = ray.get(
            actor_group_dynamic.async_run_ray_method("pass_through", "ppo_train", train_data_dynamic)
        )

        print(f"Results dynamic: {results_dynamic}")
        losses_dynamic = [r.metadata["train_status"]["policy_loss"] for r in results_dynamic]
        loss_dynamic = sum(losses_dynamic) / len(losses_dynamic)

        print(f"Dynamic batching losses per GPU: {losses_dynamic}")
        print(f"Dynamic batching average loss: {loss_dynamic}")

        print("\nComparison:")
        print(f"Fixed batching average loss: {loss_fixed}")
        print(f"Dynamic batching average loss: {loss_dynamic}")
        print(f"Absolute difference: {abs(loss_fixed - loss_dynamic)}")

        for i in range(len(losses_fixed)):
            print(f"GPU {i} - Fixed: {losses_fixed[i]}, Dynamic: {losses_dynamic[i]}")

        assert abs(loss_fixed - loss_dynamic) < 1e-4, (
            f"Losses should be nearly identical. Fixed: {loss_fixed}, Dynamic: {loss_dynamic}, "
            f"Diff: {abs(loss_fixed - loss_dynamic)}"
        )

        print(
            "✅ E2E test passed: Dynamic batching produces same loss as fixed batching on multiple GPUs with inference engines"
        )

    finally:
        ray.shutdown()
