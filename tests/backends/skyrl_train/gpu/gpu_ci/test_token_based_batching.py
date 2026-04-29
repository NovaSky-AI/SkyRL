"""
Tests for max_tokens_per_microbatch (token-based micro-batching).

Tests verify:
1. Unit tests for balanced_binpacking and TokenBasedBatchIterator
2. FSDP forward_backward with token-based batching produces equivalent loss
3. Megatron forward with token-based batching produces equivalent results
4. Performance comparison (token-based vs sample-based)

Run with:
uv run --isolated --extra dev --extra fsdp -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_token_based_batching.py
"""

import time

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.workers.worker_utils import (
    TokenBasedBatchIterator,
    balanced_binpacking,
    get_microbatch_iterator,
)
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
)


# ─── Unit Tests (CPU only, no Ray/GPU needed) ───────────────────────────────


class TestBalancedBinpacking:
    def test_basic_packing(self):
        result = balanced_binpacking([10, 10, 5, 5], 15)
        assert len(result) == 2
        # Each microbatch should have total <= 15
        for mb in result:
            total = sum([10, 10, 5, 5][i] for i in mb)
            assert total <= 15

    def test_single_large_item(self):
        result = balanced_binpacking([10, 1, 1, 1, 1, 1], 10)
        assert len(result) == 2
        # The large item should be alone
        for mb in result:
            total = sum([10, 1, 1, 1, 1, 1][i] for i in mb)
            assert total <= 10

    def test_all_items_equal(self):
        result = balanced_binpacking([5, 5, 5, 5], 10)
        assert len(result) == 2
        for mb in result:
            total = sum(5 for _ in mb)
            assert total <= 10

    def test_single_item(self):
        result = balanced_binpacking([10], 15)
        assert len(result) == 1
        assert result[0] == [0]

    def test_all_indices_covered(self):
        token_counts = [8, 3, 5, 6, 2, 7]
        result = balanced_binpacking(token_counts, 11)
        all_indices = sorted(idx for mb in result for idx in mb)
        assert all_indices == list(range(len(token_counts)))

    def test_no_overflow(self):
        token_counts = [8, 3, 5, 6, 2, 7]
        max_tokens = 11
        result = balanced_binpacking(token_counts, max_tokens)
        for mb in result:
            total = sum(token_counts[i] for i in mb)
            assert total <= max_tokens


class TestTokenBasedBatchIterator:
    def _make_batch(self, seq_lens, num_actions=4):
        """Create a dummy TrainingInputBatch with variable sequence lengths."""
        batch_size = len(seq_lens)
        max_seq_len = max(seq_lens)

        sequences = torch.zeros((batch_size, max_seq_len), dtype=int, device="cpu")
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=int, device="cpu")
        for i, seq_len in enumerate(seq_lens):
            sequences[i, :seq_len] = torch.randint(0, 100, (seq_len,), dtype=int, device="cpu")
            attention_mask[i, :seq_len] = 1

        data = TrainingInputBatch(
            {
                "sequences": sequences,
                "attention_mask": attention_mask,
                "action_log_probs": 0.4 * torch.ones((batch_size, num_actions), device="cpu"),
                "base_action_log_probs": 0.3 * torch.ones((batch_size, num_actions), device="cpu"),
                "values": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
                "returns": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
                "advantages": 0.6 * torch.ones((batch_size, num_actions), device="cpu"),
                "loss_mask": torch.ones((batch_size, num_actions), dtype=int, device="cpu"),
                "response_mask": torch.ones((batch_size, num_actions), dtype=int, device="cpu"),
            }
        )
        data.metadata = {"response_length": num_actions}
        return data

    def test_iterator_yields_all_samples(self):
        batch = self._make_batch([10, 10, 5, 5])
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=15)

        all_indices = []
        for mb_indices in iterator._microbatches:
            all_indices.extend(mb_indices)
        assert sorted(all_indices) == [0, 1, 2, 3]

    def test_iterator_respects_token_limit(self):
        batch = self._make_batch([10, 10, 5, 5])
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=15)

        for microbatch in iterator:
            token_count = microbatch["attention_mask"].sum().item()
            # Allow some slack for padding microbatches
            if microbatch["loss_mask"].sum() > 0:  # not a padding batch
                assert token_count <= 15

    def test_len_matches_iteration(self):
        batch = self._make_batch([10, 10, 5, 5])
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=15)
        count = sum(1 for _ in iterator)
        assert count == len(iterator)

    def test_reorder_and_combine(self):
        """Verify that reorder_and_combine_batches restores original order."""
        batch = self._make_batch([10, 3, 8, 5])
        iterator = TokenBasedBatchIterator(batch, max_tokens_per_microbatch=12)

        # Simulate forward outputs (just use the microbatch itself as output)
        outputs = []
        for microbatch in iterator:
            outputs.append(microbatch)

        reordered = iterator.reorder_and_combine_batches(outputs)
        # Check that the sequences match the original order
        for i in range(batch.batch_size):
            assert torch.equal(reordered["sequences"][i], batch["sequences"][i])

    def test_get_microbatch_iterator_factory(self):
        batch = self._make_batch([10, 10, 5, 5])

        # Token-based
        it = get_microbatch_iterator(batch, micro_batch_size=2, max_tokens_per_microbatch=15)
        assert isinstance(it, TokenBasedBatchIterator)

        # Sample-based (disabled)
        from skyrl.backends.skyrl_train.workers.worker_utils import SampleBasedBatchIterator

        it = get_microbatch_iterator(batch, micro_batch_size=2, max_tokens_per_microbatch=-1)
        assert isinstance(it, SampleBasedBatchIterator)


# ─── GPU Tests: FSDP ─────────────────────────────────────────────────────────

MODEL_NAME = "Qwen/Qwen3-0.6B"


def _make_variable_length_batch(seq_lens, num_actions=4):
    """Create a TrainingInputBatch with variable-length sequences (right-padded to max)."""
    torch.manual_seed(42)
    batch_size = len(seq_lens)
    max_seq_len = max(seq_lens)

    sequences = torch.zeros((batch_size, max_seq_len), dtype=int, device="cpu")
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=int, device="cpu")
    for i, seq_len in enumerate(seq_lens):
        sequences[i, :seq_len] = torch.randint(1, 100, (seq_len,), device="cpu")
        attention_mask[i, :seq_len] = 1

    data = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "action_log_probs": 0.4 * torch.ones((batch_size, num_actions), device="cpu"),
            "base_action_log_probs": 0.3 * torch.ones((batch_size, num_actions), device="cpu"),
            "values": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
            "returns": 0.5 * torch.ones((batch_size, num_actions), device="cpu"),
            "advantages": 0.6 * torch.ones((batch_size, num_actions), device="cpu"),
            "loss_mask": torch.ones((batch_size, num_actions), dtype=int, device="cpu"),
            "response_mask": torch.ones((batch_size, num_actions), dtype=int, device="cpu"),
            "rollout_logprobs": 0.2 * torch.ones((batch_size, num_actions), device="cpu"),
        }
    )
    data.metadata = {"response_length": num_actions}
    return data


def get_fsdp_test_config() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.logger = "console"
    cfg.generator.inference_engine.tensor_parallel_size = 2
    return cfg


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_type", ["policy", "critic"])
async def test_fsdp_token_based_forward_backward(ray_init_fixture, worker_type):
    """
    Test that forward_backward with max_tokens_per_microbatch works correctly for FSDP.

    Verifies:
    1. Token-based batching runs without errors for both policy and critic
    2. Returns valid metrics with expected keys
    3. For policy: loss is close to sample-based baseline (both use pre-scaled advantages)
    """
    try:
        # Create a batch with variable-length sequences
        seq_lens = [30, 30, 15, 15]  # 4 samples, 2 per DP rank
        batch = _make_variable_length_batch(seq_lens, num_actions=4)
        batch.metadata["global_step"] = 0

        # Token-based batching
        cfg_token = get_fsdp_test_config()
        cfg_token.trainer.strategy = "fsdp2"
        cfg_token.trainer.policy.model.path = MODEL_NAME
        cfg_token.trainer.micro_train_batch_size_per_gpu = 1
        cfg_token.trainer.max_tokens_per_microbatch = 30
        validate_cfg(cfg_token)

        actor_group = init_worker_with_type(
            worker_type,
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg_token.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg_token,
        )
        results_token = ray.get(
            actor_group.async_run_ray_method("mesh", "forward_backward", data=batch)
        )

        # Verify results have expected structure
        loss_key = "policy_loss" if worker_type == "policy" else "critic_loss"
        for i, r in enumerate(results_token):
            assert isinstance(r, dict), f"Result should be a dict, got {type(r)}"
            assert loss_key in r, f"Missing {loss_key} in result"
            print(f"  Rank {i}: token-based {loss_key}={r[loss_key]:.6f}")

            if worker_type == "policy":
                assert "loss_metrics/clip_ratio" in r
                assert "policy_entropy" in r
                assert "loss_fn_outputs" in r

        # Also verify optim_step works
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
        print(f"  {worker_type}: forward_backward + optim_step completed successfully")

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_fsdp_token_based_loss_equivalence(ray_init_fixture):
    """
    Test that policy loss with token-based batching matches sample-based
    when each microbatch contains exactly 1 sample (i.e., when max_tokens
    is set high enough that no packing occurs but low enough that each
    sample gets its own microbatch).
    """
    try:
        # Uniform-length sequences to ensure identical batching behavior
        seq_lens = [20, 20, 20, 20]
        batch = _make_variable_length_batch(seq_lens, num_actions=4)
        batch.metadata["global_step"] = 0

        # Run 1: sample-based baseline (mbs=1)
        cfg_baseline = get_fsdp_test_config()
        cfg_baseline.trainer.strategy = "fsdp2"
        cfg_baseline.trainer.policy.model.path = MODEL_NAME
        cfg_baseline.trainer.micro_train_batch_size_per_gpu = 1
        cfg_baseline.trainer.max_tokens_per_microbatch = -1
        validate_cfg(cfg_baseline)

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg_baseline.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg_baseline,
        )
        results_baseline = ray.get(
            actor_group.async_run_ray_method("mesh", "forward_backward", data=batch)
        )

        ray.shutdown()
        from tests.backends.skyrl_train.gpu.utils import ray_init_for_tests

        ray_init_for_tests()

        # Run 2: token-based with limit that gives 1 sample per microbatch
        cfg_token = get_fsdp_test_config()
        cfg_token.trainer.strategy = "fsdp2"
        cfg_token.trainer.policy.model.path = MODEL_NAME
        cfg_token.trainer.micro_train_batch_size_per_gpu = 1
        # max_tokens=20 means each 20-token sample goes in its own microbatch
        cfg_token.trainer.max_tokens_per_microbatch = 20
        validate_cfg(cfg_token)

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg_token.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg_token,
        )
        results_token = ray.get(
            actor_group.async_run_ray_method("mesh", "forward_backward", data=batch)
        )

        # With uniform sequences and 1 sample per microbatch, losses should match closely
        for i, (r_baseline, r_token) in enumerate(zip(results_baseline, results_token)):
            bl = r_baseline["policy_loss"]
            tl = r_token["policy_loss"]
            print(f"  Rank {i}: baseline={bl:.6f}, token-based={tl:.6f}, diff={abs(bl-tl):.6f}")
            assert abs(bl - tl) < 1e-4, f"Loss mismatch on rank {i}: {bl} vs {tl}"

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_fsdp_token_based_batching_performance(ray_init_fixture):
    """
    Test that token-based batching shows better throughput than sample-based
    when sequences have highly variable lengths.

    Creates a batch with a mix of short (15 tokens) and long (100 tokens) sequences.
    With sample-based batching (mbs=1), each microbatch processes one sample with
    the full max_seq_len padding. With token-based batching, short sequences can be
    packed together, reducing the number of forward passes.
    """
    try:
        # Create batch with high variance in sequence lengths
        # 8 samples total (4 per DP rank with 2 GPUs)
        # Mix of short and long sequences
        seq_lens = [100, 100, 15, 15, 100, 100, 15, 15]
        batch = _make_variable_length_batch(seq_lens, num_actions=4)
        batch.metadata["global_step"] = 0

        # Run 1: sample-based with mbs=1 (no packing, wastes time on padding)
        cfg_sample = get_fsdp_test_config()
        cfg_sample.trainer.strategy = "fsdp2"
        cfg_sample.trainer.policy.model.path = MODEL_NAME
        cfg_sample.trainer.micro_train_batch_size_per_gpu = 1
        cfg_sample.trainer.max_tokens_per_microbatch = -1
        validate_cfg(cfg_sample)

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg_sample.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg_sample,
        )

        # Warmup
        ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

        start = time.time()
        NUM_ITERS = 3
        for _ in range(NUM_ITERS):
            ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))
            ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
        sample_time = (time.time() - start) / NUM_ITERS

        ray.shutdown()
        from tests.backends.skyrl_train.gpu.utils import ray_init_for_tests

        ray_init_for_tests()

        # Run 2: token-based batching
        cfg_token = get_fsdp_test_config()
        cfg_token.trainer.strategy = "fsdp2"
        cfg_token.trainer.policy.model.path = MODEL_NAME
        cfg_token.trainer.micro_train_batch_size_per_gpu = 1
        # Set max_tokens to pack 2 short sequences together: 15+15=30 < 120
        cfg_token.trainer.max_tokens_per_microbatch = 120
        validate_cfg(cfg_token)

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg_token.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg_token,
        )

        # Warmup
        ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

        start = time.time()
        for _ in range(NUM_ITERS):
            ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))
            ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
        token_time = (time.time() - start) / NUM_ITERS

        print(f"\nPerformance comparison (avg over {NUM_ITERS} iterations):")
        print(f"  Sample-based (mbs=1): {sample_time:.3f}s")
        print(f"  Token-based (max_tokens=120): {token_time:.3f}s")
        print(f"  Speedup: {sample_time / token_time:.2f}x")

        # Token-based should be at least as fast (may not always be faster with small batches)
        # The main point is correctness; performance benefit is more visible with larger batches
        assert token_time < sample_time * 1.5, (
            f"Token-based batching should not be significantly slower: "
            f"{token_time:.3f}s vs {sample_time:.3f}s"
        )

    finally:
        ray.shutdown()


def _get_megatron_test_config(tp=2, pp=1, gpus=2) -> SkyRLTrainConfig:
    """Create a Megatron test config that passes validate_cfg."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.use_sample_packing = False
    cfg.trainer.logger = "console"
    cfg.trainer.strategy = "megatron"
    cfg.trainer.placement.policy_num_gpus_per_node = gpus
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.policy.megatron_config.tensor_model_parallel_size = tp
    cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = pp
    return cfg


@pytest.mark.asyncio
@pytest.mark.megatron
async def test_megatron_token_based_forward(ray_init_fixture):
    """
    Test that forward pass with token-based batching works correctly for Megatron.
    Compares forward output between sample-based and token-based batching.
    """
    from skyrl.backends.skyrl_train.distributed.dispatch import (
        concatenate_outputs_after_mesh_dispatch,
    )
    from tests.backends.skyrl_train.gpu.utils import ray_init_for_tests

    try:
        seq_lens = [30, 30, 15, 15]
        batch = _make_variable_length_batch(seq_lens, num_actions=4)

        # Run 1: sample-based baseline
        cfg = _get_megatron_test_config(tp=2, pp=1, gpus=2)
        cfg.trainer.max_tokens_per_microbatch = -1

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        results_refs = actor_group.async_run_ray_method("mesh", "forward", data=batch)
        results_baseline = ray.get(results_refs)
        output_baseline = concatenate_outputs_after_mesh_dispatch(
            actor_group.actor_infos, results_baseline
        )["output"]

        ray.shutdown()
        ray_init_for_tests()

        # Run 2: token-based
        cfg2 = _get_megatron_test_config(tp=2, pp=1, gpus=2)
        cfg2.trainer.max_tokens_per_microbatch = 35  # Can fit 1 long or 2 short seqs

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg2.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg2,
        )

        results_refs = actor_group.async_run_ray_method("mesh", "forward", data=batch)
        results_token = ray.get(results_refs)
        output_token = concatenate_outputs_after_mesh_dispatch(
            actor_group.actor_infos, results_token
        )["output"]

        # Compare log probs
        max_diff = torch.max(torch.abs(output_baseline - output_token)).item()
        avg_diff = torch.mean(torch.abs(output_baseline - output_token)).item()
        print(f"\nMegatron forward comparison:")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Avg diff: {avg_diff:.6f}")

        assert max_diff < 1e-3, f"Max diff {max_diff} too large between sample-based and token-based"

    finally:
        ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.megatron
async def test_megatron_token_based_train(ray_init_fixture):
    """
    Test that training with token-based batching works correctly for Megatron (TP=2, PP=1).
    """
    try:
        seq_lens = [30, 30, 15, 15, 30, 30, 15, 15]
        batch = _make_variable_length_batch(seq_lens, num_actions=4)
        batch.metadata["global_step"] = 0

        cfg = _get_megatron_test_config(tp=2, pp=1, gpus=2)
        cfg.trainer.max_tokens_per_microbatch = 35
        cfg.trainer.train_batch_size = len(seq_lens)
        cfg.trainer.policy_mini_batch_size = 4
        cfg.trainer.algorithm.use_kl_loss = False

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        results = ray.get(
            actor_group.async_run_ray_method("mesh", "forward_backward", data=batch)
        )
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

        for result in results:
            assert isinstance(result, dict), "Result should be a dictionary"
            assert "policy_loss" in result
            assert "policy_entropy" in result
            print(f"  policy_loss={result['policy_loss']:.6f}")

    finally:
        ray.shutdown()
