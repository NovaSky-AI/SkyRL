"""
Tests for max_tokens_per_microbatch (token-based micro-batching).

Tests verify:
1. FSDP forward_backward with token-based batching produces equivalent loss
2. Megatron forward with token-based batching produces equivalent results
3. Performance comparison (token-based vs sample-based)

Unit tests for balanced_binpacking and TokenBasedBatchIterator (CPU only) live in
tests/backends/skyrl_train/test_token_based_batching_utils.py

Run with:
uv run --isolated --extra dev --extra megatron -- pytest -s tests/backends/skyrl_train/gpu/gpu_ci/test_token_based_batching.py
"""

import time

import pytest
import ray
import torch

from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    init_worker_with_type,
)

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
@pytest.mark.parametrize("worker_type", ["policy"])
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
        results_token = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))

        # Verify results have expected structure
        loss_key = "policy_loss" if worker_type == "policy" else "critic_loss"
        for i, r in enumerate(results_token):
            assert isinstance(r, WorkerOutput), f"Result should be a WorkerOutput, got {type(r)}"
            assert loss_key in r.metrics, f"Missing {loss_key} in result metrics"
            print(f"  Rank {i}: token-based {loss_key}={r.metrics[loss_key]:.6f}")

            if worker_type == "policy":
                assert "loss_metrics/clip_ratio" in r.metrics
                assert "policy_entropy" in r.metrics
                assert len(r.loss_fn_outputs) > 0

            # Token-based microbatch-count diagnostics (gated on max_tokens_per_microbatch > 0).
            # 2 GPUs -> DP=2; with seq_lens=[30,30,15,15] @ max_tokens=30 the ranks bin-pack
            # into uneven microbatch counts, so the short rank gets padding microbatches.
            assert "num_microbatches" in r.metrics, "missing num_microbatches metric"
            assert "num_padding_microbatches" in r.metrics, "missing num_padding_microbatches metric"
            assert r.metrics["num_microbatches"] > 0
            assert r.metrics["num_padding_microbatches"] > 0, "expected padding microbatches at DP=2"

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

    This also exercises the padding-microbatch path at DP>1: with 2 GPUs the
    batch splits contiguously into rank0=[20,5] (-> 2 microbatches at max_tokens=20)
    and rank1=[10,5] (-> 1 microbatch), so rank1 gets a loss-neutral padding
    microbatch and the loss must still match the sample-based baseline.
    """
    try:
        # Uniform-length sequences to ensure identical batching behavior
        seq_lens = [20, 5, 10, 5]
        batch = _make_variable_length_batch(seq_lens, num_actions=4)
        batch.metadata["global_step"] = 0

        # Run 1: sample-based baseline (mbs=1)
        cfg_baseline = get_fsdp_test_config()
        cfg_baseline.trainer.strategy = "fsdp2"
        cfg_baseline.trainer.policy.model.path = MODEL_NAME
        cfg_baseline.trainer.micro_train_batch_size_per_gpu = 1
        cfg_baseline.trainer.max_tokens_per_microbatch = -1
        cfg_baseline.trainer.algorithm.use_kl_loss = False
        validate_cfg(cfg_baseline)

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg_baseline.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg_baseline,
        )
        results_baseline = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))

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
        cfg_token.trainer.algorithm.use_kl_loss = False
        validate_cfg(cfg_token)

        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg_token.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg_token,
        )
        results_token = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))

        # With uniform sequences and 1 sample per microbatch, losses should match closely
        for i, (r_baseline, r_token) in enumerate(zip(results_baseline, results_token)):
            bl = r_baseline.metrics["policy_loss"]
            tl = r_token.metrics["policy_loss"]
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
            f"Token-based batching should not be significantly slower: " f"{token_time:.3f}s vs {sample_time:.3f}s"
        )

    finally:
        ray.shutdown()


def _get_megatron_test_config(tp=2, pp=1, gpus=2) -> SkyRLTrainConfig:
    """Create a Megatron test config that passes validate_cfg."""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.micro_forward_batch_size_per_gpu = 2
    cfg.trainer.micro_train_batch_size_per_gpu = 2
    cfg.trainer.remove_microbatch_padding = False
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
        loss_fn_outputs_to_tensor,
    )
    from tests.backends.skyrl_train.gpu.utils import ray_init_for_tests

    try:
        seq_lens = [30, 30, 15, 15]
        batch = _make_variable_length_batch(seq_lens, num_actions=4)

        # Run 1: sample-based baseline
        cfg = _get_megatron_test_config(tp=2, pp=2, gpus=4)
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
        baseline_output = WorkerOutput.cat(actor_group.actor_infos, results_baseline)
        output_baseline = loss_fn_outputs_to_tensor(baseline_output.loss_fn_outputs, key="logprobs")

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
        token_output = WorkerOutput.cat(actor_group.actor_infos, results_token)
        output_token = loss_fn_outputs_to_tensor(token_output.loss_fn_outputs, key="logprobs")

        # Compare log probs
        max_diff = torch.max(torch.abs(output_baseline - output_token)).item()
        avg_diff = torch.mean(torch.abs(output_baseline - output_token)).item()
        print("\nMegatron forward comparison:")
        print(f"  Max diff: {max_diff:.6f}")
        print(f"  Avg diff: {avg_diff:.6f}")

        assert max_diff < 1e-3, f"Max diff {max_diff} too large between sample-based and token-based"

    finally:
        ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.megatron
@pytest.mark.parametrize("remove_microbatch_padding", [True, False])
@pytest.mark.parametrize(
    "tp, pp, gpus, seq_lens, max_tokens",
    [
        # DP=1 (PP=2): varied/odd lengths so token-based packing yields uneven
        # microbatch sizes (the longest and an unpaired short each land alone and
        # get a dummy-padded partner). Odd lengths (31, 21) keep the packed/THD
        # fp noise visible. No padding *microbatches* here (DP=1).
        (2, 2, 4, [50, 40, 31, 30, 20, 21, 10, 10], 55),
        # DP=2 (PP=1): exercises the padding-*microbatch* path. Mesh dispatch
        # chunks the batch into two contiguous halves, so with max_tokens=30:
        #   rank0 = [30,30,30,30] -> 4 microbatches
        #   rank1 = [15,15,15,15] -> 2 microbatches (+2 padding microbatches)
        # Padding microbatches only appear when DP ranks bin-pack into different
        # microbatch counts, i.e. DP > 1.
        (2, 1, 4, [30, 30, 30, 30, 15, 15, 15, 15], 30),
    ],
    ids=["dp1_pp2", "dp2_pp1_padding_microbatch"],
)
async def test_megatron_token_based_loss_equivalence(
    ray_init_fixture, tp, pp, gpus, seq_lens, max_tokens, remove_microbatch_padding
):
    """
    Test that the policy loss from forward_backward matches between sample-based
    and token-based batching for Megatron.

    The loss is normalized over the full mini-batch, so packing samples into
    token-based microbatches (rather than fixed-size sample-based microbatches)
    should produce an equivalent loss up to numerical tolerance.

    Parametrized over two axes:

    - ``remove_microbatch_padding`` covers both forward paths. Token-based batching
      pads uneven microbatches with dummy (single-token) rows, and (at DP>1) appends
      loss-neutral padding microbatches; both are held to different tolerances:

      * dense (``False``): each row is attended independently at full width, so
        real-sample logits are bitwise-stable regardless of dummy rows / grouping.
        We expect ~exact equivalence (``1e-6``).
      * packed/THD (``True``): dummy rows pack as length-1 segments, shifting the
        ``cu_seqlens`` layout vs the sample-based grouping. Block-diagonal attention
        keeps this mathematically equivalent, but the varlen kernel's fp reduction
        order shifts, giving ~1e-4 noise. Held to ``1e-3``.

    - topology covers ``DP=1`` (PP=2) and ``DP=2`` (PP=1). The DP=2 case is the
      regression guard for padding microbatches (only emitted when DP ranks
      bin-pack into different microbatch counts).
    """
    from tests.backends.skyrl_train.gpu.utils import ray_init_for_tests

    try:

        def _make_cfg(max_tokens_per_microbatch):
            cfg = _get_megatron_test_config(tp=tp, pp=pp, gpus=gpus)
            cfg.trainer.max_tokens_per_microbatch = max_tokens_per_microbatch
            cfg.trainer.remove_microbatch_padding = remove_microbatch_padding
            cfg.trainer.train_batch_size = len(seq_lens)
            cfg.trainer.policy_mini_batch_size = len(seq_lens)
            cfg.trainer.algorithm.use_kl_loss = False
            return cfg

        # Run 1: sample-based baseline
        batch = _make_variable_length_batch(seq_lens, num_actions=4)
        batch.metadata["global_step"] = 0
        cfg_baseline = _make_cfg(max_tokens_per_microbatch=-1)
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg_baseline.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg_baseline,
        )
        results_baseline = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))

        ray.shutdown()
        ray_init_for_tests()

        # Run 2: token-based (packs short seqs together; long/unpaired seqs get their
        # own dummy-padded microbatch, and at DP>1 short ranks get padding microbatches)
        batch = _make_variable_length_batch(seq_lens, num_actions=4)
        batch.metadata["global_step"] = 0
        cfg_token = _make_cfg(max_tokens_per_microbatch=max_tokens)
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg_token.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg_token,
        )
        results_token = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))

        # Token-based batching exposes microbatch-count diagnostics. They are gated on
        # max_tokens_per_microbatch > 0, so the sample-based baseline must not have them.
        dp_size = gpus // (tp * pp)
        for r in results_baseline:
            assert "num_microbatches" not in r.metrics
            assert "num_padding_microbatches" not in r.metrics
        for r in results_token:
            assert "num_microbatches" in r.metrics, "missing num_microbatches metric"
            assert "num_padding_microbatches" in r.metrics, "missing num_padding_microbatches metric"
            assert r.metrics["num_microbatches"] > 0
            if dp_size > 1:
                # Uneven bin-packing across DP ranks -> some ranks get padding microbatches
                # (averaged across DP, so a positive fractional value).
                assert (
                    r.metrics["num_padding_microbatches"] > 0
                ), f"expected padding microbatches at DP={dp_size}, got {r.metrics['num_padding_microbatches']}"
            else:
                assert r.metrics["num_padding_microbatches"] == 0, "no padding microbatches expected at DP=1"
        print(
            f"  num_microbatches={results_token[0].metrics['num_microbatches']}, "
            f"num_padding_microbatches={results_token[0].metrics['num_padding_microbatches']}"
        )

        # Dense attention is bitwise-stable across groupings; packed/THD has
        # ~1e-4 varlen-kernel fp noise from the shifted cu_seqlens layout.
        tol = 1e-3 if remove_microbatch_padding else 1e-6
        print(f"\nMegatron loss equivalence (remove_microbatch_padding={remove_microbatch_padding}, tol={tol}):")
        for i, (r_baseline, r_token) in enumerate(zip(results_baseline, results_token)):
            bl = r_baseline.metrics["policy_loss"]
            tl = r_token.metrics["policy_loss"]
            print(f"  Rank {i}: baseline={bl:.6f}, token-based={tl:.6f}, diff={abs(bl - tl):.6f}")
            assert abs(bl - tl) < tol, f"Loss mismatch on rank {i}: {bl} vs {tl} (tol={tol})"

    finally:
        ray.shutdown()
