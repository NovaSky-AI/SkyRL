"""
Tests for automatic micro-batch sizing via token-budget profiling.

Run with:
    export NCCL_P2P_DISABLE=1
    export NCCL_SHM_DISABLE=1
    uv run --isolated --extra dev -- pytest tests/gpu/gpu_ci/test_auto_microbatch.py -v -s

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
    """Build a SkyRLConfig with `auto_micro_batch_size` enabled."""
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
async def test_auto_token_budget_full_shard(ray_init_fixture, strategy):
    """2 GPUs, model fully sharded — all workers must agree on the token budget."""
    cfg = get_auto_test_config(num_gpus=2, fsdp_size=-1, strategy=strategy)

    try:
        actor_group = init_worker_with_type(
            "policy",
            num_gpus_per_node=2,
            cfg=cfg,
        )

        max_seq_len = cfg.trainer.max_prompt_length + cfg.generator.sampling_params.max_generate_length

        # token budget profiling on all workers
        results = ray.get(
            actor_group.async_run_ray_method(
                "pass_through",
                "auto_determine_token_budget",
                max_seq_len,
            )
        )

        # all workers should return the same value (coordinated via all_reduce)
        assert all(r == results[0] for r in results), f"Workers disagree: {results}"
        assert results[0] > 0, f"Token budget must be positive, got {results[0]}"

        # token budget should be at least max_seq_len (can fit 1 sample)
        assert results[0] >= max_seq_len, f"Token budget {results[0]} is less than max_seq_len {max_seq_len}"

        print(f"[{strategy}] Token budget: {results[0]}")
    finally:
        ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.skipif(
    torch.cuda.device_count() < 4,
    reason="Need 4 GPUs for HYBRID_SHARD test",
)
async def test_auto_token_budget_hybrid_shard(ray_init_fixture):
    """4 GPUs, fsdp_size=2 — HYBRID_SHARD with 2 FSDP groups of 2."""
    cfg = get_auto_test_config(num_gpus=4, fsdp_size=2, strategy="fsdp2")

    try:
        actor_group = init_worker_with_type(
            "policy",
            num_gpus_per_node=4,
            cfg=cfg,
        )

        max_seq_len = cfg.trainer.max_prompt_length + cfg.generator.sampling_params.max_generate_length

        results = ray.get(
            actor_group.async_run_ray_method(
                "pass_through",
                "auto_determine_token_budget",
                max_seq_len,
            )
        )

        assert all(r == results[0] for r in results), f"Workers disagree: {results}"
        assert results[0] > 0

        print(f"[hybrid] Token budget: {results[0]}")
    finally:
        ray.shutdown()


def test_memory_aware_iterator_packing():
    """MemoryAwareBatchIterator should pack short sequences more densely."""
    from skyrl_train.workers.worker_utils import MemoryAwareBatchIterator
    from skyrl_train.training_batch import TrainingInputBatch

    batch_size = 6
    padded_len = 100
    actual_lens = [100, 80, 60, 40, 20, 10]

    sequences = torch.randint(0, 100, (batch_size, padded_len))
    attention_mask = torch.zeros(batch_size, padded_len, dtype=torch.long)
    for i, length in enumerate(actual_lens):
        attention_mask[i, :length] = 1

    data = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
        }
    )
    data.metadata = {"response_length": padded_len}

    iterator = MemoryAwareBatchIterator(data, token_budget=200)

    micro_batches = list(iterator)
    assert len(micro_batches) == 3, f"Expected 3 micro-batches, got {len(micro_batches)}"

    assert micro_batches[0].sequences.shape[0] == 2
    assert micro_batches[1].sequences.shape[0] == 3
    assert micro_batches[2].sequences.shape[0] == 1
    total = sum(mb.sequences.shape[0] for mb in micro_batches)
    assert total == batch_size


def test_memory_aware_iterator_all_same_length():
    """When all sequences are the same length, packing should be uniform."""
    from skyrl_train.workers.worker_utils import MemoryAwareBatchIterator
    from skyrl_train.training_batch import TrainingInputBatch

    batch_size = 8
    seq_len = 50

    data = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }
    )
    data.metadata = {"response_length": seq_len}

    iterator = MemoryAwareBatchIterator(data, token_budget=200)
    micro_batches = list(iterator)
    assert len(micro_batches) == 2
    assert micro_batches[0].sequences.shape[0] == 4
    assert micro_batches[1].sequences.shape[0] == 4


def test_memory_aware_iterator_single_sequence():
    """A batch with a single very long sequence should produce 1 micro-batch."""
    from skyrl_train.workers.worker_utils import MemoryAwareBatchIterator
    from skyrl_train.training_batch import TrainingInputBatch

    data = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (1, 500)),
            "attention_mask": torch.ones(1, 500, dtype=torch.long),
        }
    )
    data.metadata = {"response_length": 500}

    iterator = MemoryAwareBatchIterator(data, token_budget=1000)
    micro_batches = list(iterator)
    assert len(micro_batches) == 1
    assert micro_batches[0].sequences.shape[0] == 1


def test_memory_aware_iterator_budget_too_small():
    """If budget < max_seq_len, each sequence gets its own micro-batch."""
    from skyrl_train.workers.worker_utils import MemoryAwareBatchIterator
    from skyrl_train.training_batch import TrainingInputBatch

    batch_size = 3
    seq_len = 100

    data = TrainingInputBatch(
        {
            "sequences": torch.randint(0, 100, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        }
    )
    data.metadata = {"response_length": seq_len}

    iterator = MemoryAwareBatchIterator(data, token_budget=50)
    micro_batches = list(iterator)
    assert len(micro_batches) == batch_size
