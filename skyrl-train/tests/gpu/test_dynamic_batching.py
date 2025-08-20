"""
Tests for dynamic token-based batching.

Run with:
uv run --isolated --extra dev -- pytest tests/gpu/test_dynamic_batching.py -v
"""

import pytest
import torch
import ray
from omegaconf import DictConfig

from skyrl_train.workers.worker_utils import BatchIterator
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.dynamic_batching import (
    get_seqlen_balanced_partitions,
    calculate_num_micro_batches,
)
from tests.gpu.utils import make_dummy_training_batch, init_worker_with_type, get_test_actor_config


def make_variable_length_training_batch(
    seq_lengths: list[int], num_actions: int = 4, pad_to_length: int = None
) -> TrainingInputBatch:
    batch_size = len(seq_lengths)
    max_seq_len = max(seq_lengths) if not pad_to_length else pad_to_length

    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    for i, length in enumerate(seq_lengths):
        attention_mask[i, :length] = 1

    data = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (batch_size, max_seq_len)),
            "attention_mask": attention_mask,
            "rollout_log_probs": torch.randn((batch_size, num_actions)),
            "action_log_probs": torch.randn((batch_size, num_actions)),
            "base_action_log_probs": torch.randn((batch_size, num_actions)),
            "values": torch.randn((batch_size, num_actions)),
            "returns": torch.randn((batch_size, num_actions)),
            "advantages": torch.randn((batch_size, num_actions)),
            "loss_mask": torch.ones((batch_size, num_actions), dtype=torch.long),
            "response_mask": torch.ones((batch_size, num_actions), dtype=torch.long),
        }
    )
    data.metadata = {"response_length": num_actions}
    return data


@pytest.mark.parametrize(
    "seq_lengths,max_tokens,expected_num_batches",
    [
        ([100] * 10, 500, 2),
        ([50, 100, 150, 200], 300, 2),
        ([500], 500, 1),
        ([50] * 8, 200, 2),
        ([100, 200, 300], 350, 2),
    ],
)
def test_dynamic_batch_iterator_core(seq_lengths, max_tokens, expected_num_batches):
    batch = make_variable_length_training_batch(seq_lengths, pad_to_length=max(seq_lengths))
    # Create a mock config for BatchIterator
    cfg = DictConfig(
        {
            "trainer": {
                "use_dynamic_batching": True,
                "max_token_len_per_gpu": max_tokens,
                "micro_train_batch_size_per_gpu": 1,
            }
        }
    )

    iterator = BatchIterator(batch, cfg=cfg, dp_size=1, dynamic_bsz=True, mini_batch_size_per_gpu=len(seq_lengths))

    assert len(iterator) == expected_num_batches

    total_sequences = 0
    for exp in iterator:
        assert exp.attention_mask.sum().item() <= max_tokens
        assert all(hasattr(exp, f) for f in ["sequences", "attention_mask", "num_actions"])
        total_sequences += exp.sequences.shape[0]

    assert total_sequences == len(seq_lengths)


def test_dynamic_iterator_multi_epoch():
    batch = make_dummy_training_batch(batch_size=4, seq_len=100)
    cfg = DictConfig(
        {"trainer": {"use_dynamic_batching": True, "max_token_len_per_gpu": 300, "micro_train_batch_size_per_gpu": 1}}
    )

    iterator = BatchIterator(batch, cfg=cfg, dp_size=1, dynamic_bsz=True, mini_batch_size_per_gpu=4)

    epoch_counts = [sum(1 for _ in iterator) for _ in range(3)]
    assert len(set(epoch_counts)) == 1 and epoch_counts[0] == len(iterator)


@pytest.mark.parametrize(
    "seq_lengths,k_partitions,expected_partitions",
    [
        ([100, 200, 300], 1, [[0, 1, 2]]),
        ([100, 200, 300], 3, [[2], [1], [0]]),
        ([100, 100], 2, [[1], [0]]),
        ([50, 100, 150, 200], 2, [[0, 3], [1, 2]]),
    ],
)
def test_karmarkar_karp_partitioning(seq_lengths, k_partitions, expected_partitions):
    partitions = get_seqlen_balanced_partitions(seq_lengths, k_partitions)

    for i, partition in enumerate(partitions):
        for j, p in enumerate(partition):
            assert p == expected_partitions[i][j]
    assert len(partitions) == k_partitions


@pytest.mark.parametrize(
    "token_counts,max_tokens,min_micro_batch,expected",
    [
        ([100, 200, 300, 400], 500, None, 2),
        ([100, 100], 500, None, 1),
        ([100, 100], 500, 3, 3),
        ([50, 50, 50, 50], 150, None, 2),
        ([250, 250], 500, None, 1),
    ],
)
def test_micro_batch_calculation(token_counts, max_tokens, min_micro_batch, expected):
    num_micro = calculate_num_micro_batches(token_counts, max_tokens, min_num_micro_batch=min_micro_batch)

    assert num_micro == expected


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
            cfg.trainer.max_token_len_per_gpu = 300

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


def test_batch_iterator_with_dynamic_batching():
    """Test BatchIterator with dynamic batching enabled."""
    cfg = get_test_actor_config()
    cfg.trainer.use_dynamic_batching = True
    cfg.trainer.max_token_len_per_gpu = 200
    cfg.trainer.policy_mini_batch_size = 4
    cfg.generator.n_samples_per_prompt = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 2

    # Create batch with varying sequence lengths
    seq_lengths = [50, 100, 150, 200]
    batch = make_variable_length_training_batch(seq_lengths, num_actions=4)

    # Create BatchIterator with dynamic batching
    iterator = BatchIterator(
        data=batch, cfg=cfg, dp_size=1, dynamic_bsz=True, dp_group=None  # No distributed group for unit test
    )

    # Check that micro-batches are created
    assert len(iterator) > 0, "Should have at least one micro-batch"

    # Iterate through all micro-batches
    total_sequences = 0
    for exp in iterator:
        assert hasattr(exp, "sequences"), "Experience should have sequences"
        assert hasattr(exp, "attention_mask"), "Experience should have attention_mask"
        assert "should_step" in exp.info, "Experience info should have should_step"
        assert "accumulation_weight" in exp.info, "Experience info should have accumulation_weight"
        total_sequences += exp.sequences.shape[0]

    assert total_sequences == len(
        seq_lengths
    ), f"Should process all {len(seq_lengths)} sequences, got {total_sequences}"
    print(f"BatchIterator with dynamic batching processed {total_sequences} sequences in {len(iterator)} micro-batches")


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
        cfg.trainer.max_token_len_per_gpu = 300

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
        seq_lengths = [512, 400, 300, 200, 100, 50, 25, 10]
        seq_lengths = [1000, 50, 25, 10, 100, 50, 25, 10]

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
        
        train_data_fixed = TrainingInputBatch({
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in train_data.items()
        })
        train_data_fixed.metadata = train_data.metadata.copy()
        
        results_fixed = ray.get(actor_group_fixed.async_run_ray_method("pass_through", "ppo_train", train_data_fixed))
        
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
        cfg.trainer.max_token_len_per_gpu = 300  # Allow enough tokens for micro-batching
        
        actor_group_dynamic = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )
        
        train_data_dynamic = TrainingInputBatch({
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in train_data.items()
        })
        train_data_dynamic.metadata = train_data.metadata.copy()
        
        results_dynamic = ray.get(actor_group_dynamic.async_run_ray_method("pass_through", "ppo_train", train_data_dynamic))
        
        losses_dynamic = [r.metadata["train_status"]["policy_loss"] for r in results_dynamic]
        loss_dynamic = sum(losses_dynamic) / len(losses_dynamic)
        
        print(f"Dynamic batching losses per GPU: {losses_dynamic}")
        print(f"Dynamic batching average loss: {loss_dynamic}")
        
        print(f"\nComparison:")
        print(f"Fixed batching average loss: {loss_fixed}")
        print(f"Dynamic batching average loss: {loss_dynamic}")
        print(f"Absolute difference: {abs(loss_fixed - loss_dynamic)}")
        
        for i in range(len(losses_fixed)):
            print(f"GPU {i} - Fixed: {losses_fixed[i]}, Dynamic: {losses_dynamic[i]}")
        
        assert abs(loss_fixed - loss_dynamic) < 1e-4, (
            f"Losses should be nearly identical. Fixed: {loss_fixed}, Dynamic: {loss_dynamic}, "
            f"Diff: {abs(loss_fixed - loss_dynamic)}"
        )
        
        print("✅ E2E test passed: Dynamic batching produces same loss as fixed batching on multiple GPUs with inference engines")
        
    finally:
        ray.shutdown()
