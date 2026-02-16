"""
Tests for automatic micro-batch sizing via token-budget profiling.

Run with:
    export NCCL_P2P_DISABLE=1
    export NCCL_SHM_DISABLE=1
    uv run --isolated --extra dev -- pytest tests/gpu/gpu_ci/test_auto_microbatch.py -v -s

Requires: 2 GPUs minimum.
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
@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="Need 2 GPUs for FULL_SHARD test",
)
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


def test_memory_aware_iterator_left_padded():
    """Left-padded prompts: trimming must use rightmost position, not token count.

    Layout per the preprocessing convention:
        | [PAD] ... prompt ... | response ... [PAD] |
    The attention_mask has 1s only for actual prompt+response tokens.
    """
    from skyrl_train.workers.worker_utils import MemoryAwareBatchIterator
    from skyrl_train.training_batch import TrainingInputBatch

    max_prompt = 20
    max_response = 30
    total_len = max_prompt + max_response  # 50

    # Sequence A: prompt 15 tokens, response 25 tokens
    #   [PAD*5, prompt*15, response*25, PAD*5]
    # Sequence B: prompt 10 tokens, response 10 tokens
    #   [PAD*10, prompt*10, response*10, PAD*20]
    batch_size = 2
    sequences = torch.randint(0, 100, (batch_size, total_len))
    attention_mask = torch.zeros(batch_size, total_len, dtype=torch.long)
    loss_mask = torch.zeros(batch_size, max_response, dtype=torch.long)

    # Seq A: actual tokens at positions 5..44 (prompt 5..19, response 20..44)
    attention_mask[0, 5:45] = 1
    loss_mask[0, :25] = 1  # response length = 25

    # Seq B: actual tokens at positions 10..29 (prompt 10..19, response 20..29)
    attention_mask[1, 10:30] = 1
    loss_mask[1, :10] = 1  # response length = 10

    data = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }
    )
    data.metadata = {"response_length": max_response}

    # Budget large enough to fit both in one micro-batch
    iterator = MemoryAwareBatchIterator(data, token_budget=200)
    micro_batches = list(iterator)

    assert len(micro_batches) == 1
    mb = micro_batches[0]

    # right_trim = 50 - 45 = 5  (rightmost 1 at position 44 → effective width 45)
    # left_trim  = min(5, 10) = 5  (Seq A first 1 at pos 5, Seq B at pos 10)
    # Total-width tensors: [:, 5:45] → width 40
    # Response-width tensors: [:, :-5] → width 25 (left-trim not applied)
    assert mb.sequences.shape[1] == 40, f"Expected total width 40, got {mb.sequences.shape[1]}"
    assert mb.attention_mask.shape[1] == 40

    # Response tensor trimmed by right_trim only: 30 - 5 = 25
    assert mb.loss_mask.shape[1] == 25, f"Expected resp width 25, got {mb.loss_mask.shape[1]}"

    # Metadata must match trimmed response width
    assert mb.metadata["response_length"] == 25

    # Verify no actual tokens were lost.
    # After left-trim of 5, original positions shift: new_col = old_col - 5
    # Seq A rightmost token was at position 44 → now at column 39
    assert mb.attention_mask[0, 39] == 1 or mb.attention_mask[1, 39] == 1
    # Verify seq B's actual tokens are all preserved
    # (Seq B: original positions 10..29 → shifted to 5..24)
    for row in range(2):
        if mb.attention_mask[row].sum() == 20:  # seq B has 20 actual tokens
            assert mb.attention_mask[row, 5:25].sum() == 20
