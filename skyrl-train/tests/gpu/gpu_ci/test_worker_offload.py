"""
uv run --extra dev --isolated pytest tests/gpu/gpu_ci/test_worker_offload.py
"""

import ray
import pytest
import os
import shutil

from tests.gpu.utils import init_worker_with_type, make_dummy_training_batch, make_dummy_tensorbatch, get_rank_0_memory
from skyrl_train.config import SkyRLConfig
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.training_batch import TrainingOutputBatch

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_actor_config() -> SkyRLConfig:
    cfg = SkyRLConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.use_sample_packing = False
    cfg.trainer.logger = "console"
    cfg.generator.inference_engine_tensor_parallel_size = 2

    validate_cfg(cfg)

    return cfg


@pytest.fixture
def cfg() -> SkyRLConfig:
    return get_test_actor_config()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_type", "strategy"),
    [
        ("policy", "fsdp"),
        ("critic", "fsdp"),
        ("policy", "fsdp2"),
        ("critic", "fsdp2"),
    ],
    ids=[
        "fsdp_policy",
        "fsdp_critic",
        "fsdp2_policy",
        "fsdp2_critic",
    ],
)
async def test_critic_policy_offload_memory_and_correctness(ray_init_fixture, cfg, worker_type, strategy):
    """
    Test that offloading model memory to cpu lowers memory usage and that correctness
    is maintained after backloading and running a training step.

    steps:
    1. Initialize actor group with the specified worker class.
    2. Offload model to CPU and check memory usage.
    3. Backload model to GPU and check memory usage.
    4. Run a training step with dummy experience (with optimizer step)
    5. Offload model to CPU again and check memory usage.
    6. Backload model to GPU and check memory usage.
    7. Run another training step and ensure output consistency.

    Note for FSDP/FSDP2: optimizer is lazily initialized on the first step currently (see: https://github.com/volcengine/verl/pull/1349)
    so memory after training step + offload might be higher than after initial offload.
    """
    cfg.trainer.strategy = strategy
    # 0 learning rate and wd so we can optimizer step to free gradients but still check results are the same
    getattr(cfg.trainer, worker_type).optimizer_config.lr = 0
    getattr(cfg.trainer, worker_type).optimizer_config.weight_decay = 0
    try:
        actor_group = init_worker_with_type(
            worker_type,
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )
        get_rank_0_memory(actor_group, "After init")
        # offload then backload first (no training step)
        actor_group.offload_to_cpu()

        initial_offload_mem = get_rank_0_memory(actor_group, "After initial offload")

        # Backload to GPU
        actor_group.backload_to_gpu()
        get_rank_0_memory(actor_group, "Before training")

        dp_size = actor_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)
        # Run first forward_backward + optim_step to get optimizer initialized and stepped
        results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=dummy_batch))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

        after_training = get_rank_0_memory(actor_group, "After training")

        # Offload model to CPU
        actor_group.offload_to_cpu(offload_optimizer=True, offload_model=False)
        after_offload_optimizer = get_rank_0_memory(actor_group, "After optimizer offload")

        assert (
            after_offload_optimizer < after_training
        ), f"Memory after offload optimizer should be less than after training: {after_offload_optimizer} bytes, after training: {after_training} bytes"

        actor_group.offload_to_cpu(offload_optimizer=False, offload_model=True)
        after_offload = get_rank_0_memory(actor_group, "After model offload")

        assert (
            after_offload < after_offload_optimizer
        ), f"Memory after offload model should be less than after offload optimizer: {after_offload} bytes, after offload optimizer: {after_offload_optimizer} bytes"

        # check that allocated memory is similar to initial offload memory
        delta = abs(initial_offload_mem - after_offload)
        assert (
            delta < 4e8  # 400MB (should be close to 0 diff)
        ), f"Memory after training step + offload is not similar to initial offloaded memory: {delta} bytes. Initial offload mem: {initial_offload_mem}, after offload mem: {after_offload} bytes"

        # also check that allocated memory goes down after offloading
        delta_forward = after_training - after_offload
        assert (
            delta_forward > 0
        ), f"Memory after offloading should be less than after forward pass: {delta_forward} bytes"

        # Backload model to GPU
        actor_group.backload_to_gpu(backload_optimizer=True, backload_model=False)
        after_backload_optimizer = get_rank_0_memory(actor_group, "After backload optimizer")
        assert (
            after_backload_optimizer > after_offload
        ), f"Memory after backload optimizer should be greater than after offload: {after_backload_optimizer} bytes, after offload: {after_offload} bytes"

        actor_group.backload_to_gpu(backload_optimizer=False, backload_model=True)
        after_backload = get_rank_0_memory(actor_group, "After backload model")
        assert (
            after_backload > after_backload_optimizer
        ), f"Memory after backload model should be greater than after backload optimizer: {after_backload} bytes, after backload optimizer: {after_backload_optimizer} bytes"

        # Run training again and ensure output consistency
        results_backload = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=dummy_batch))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

        for i, result in enumerate(results):
            result_backload = results_backload[i]
            for k, v in result.items():
                assert k in result_backload
                assert v == result_backload[k], f"Results mismatch for {k}: {v} != {result_backload[k]}"

    finally:
        ray.shutdown()  # Clean up Ray resources after the test


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_type", "strategy"),
    [
        # TODO (erictang000): fsdp 1 manual offloading is broken for single gpu ref worker
        ("ref", "fsdp"),
        ("ref", "fsdp2"),
        # TODO (erictang000): Add support for reward worker.
    ],
    ids=[
        "fsdp_ref",
        "fsdp2_ref",
    ],
)
async def test_fsdp_ref_offload_memory_and_correctness(ray_init_fixture, cfg, worker_type, strategy):
    """
    Test that offloading model memory to cpu lowers memory usage and that correctness
    is maintained after backloading and running a forward pass.

    steps:
    1. Initialize actor group with the specified worker class.
    2. Offload model to CPU and check memory usage.
    3. Backload model to GPU and check memory usage.
    4. Run a forward pass with dummy experience.
    5. Offload model to CPU again and check memory usage.
    6. Backload model to GPU and check memory usage.
    7. Run another forward pass and ensure output consistency.
    """
    cfg.trainer.strategy = strategy
    # test that things work without any offloading setup by FSDP/FSDP2
    cfg.trainer.ref.fsdp_config.cpu_offload = False
    try:
        actor_group = init_worker_with_type(
            worker_type,
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )
        get_rank_0_memory(actor_group, "After init")
        # offload then backload first (no training step)
        actor_group.offload_to_cpu()
        initial_offload_mem = get_rank_0_memory(actor_group, "After initial offload")

        # should be close to 0
        assert (
            initial_offload_mem < 1e8
        ), f"Memory after offloading should be close to 0: instead {initial_offload_mem} bytes"

        # Backload to GPU
        actor_group.backload_to_gpu()
        get_rank_0_memory(actor_group, "Before forward")

        dummy_batch = make_dummy_tensorbatch()
        # Run forward pass
        results: TrainingOutputBatch = actor_group.run_method("pass_through", "forward", dummy_batch)

        after_forward = get_rank_0_memory(actor_group, "After forward")

        # Offload model to CPU
        actor_group.offload_to_cpu()

        after_offload = get_rank_0_memory(actor_group, "After offload")

        # check that allocated memory is similar to initial offload memory
        delta = abs(initial_offload_mem - after_offload)
        assert (
            delta < 1e8  # 100MB (should be close to 0 diff)
        ), f"Memory after training step + offload is not similar to initial offloaded memory: {delta} bytes"

        # also check that allocated memory goes down after offloading
        delta_forward = after_forward - after_offload
        assert (
            delta_forward > 0
        ), f"Memory after offloading should be less than after forward pass: {delta_forward} bytes"

        # Backload model to GPU
        actor_group.backload_to_gpu()

        get_rank_0_memory(actor_group, "After backload")

        # Run forward again and ensure output consistency
        results_backload: TrainingOutputBatch = actor_group.run_method("pass_through", "forward", dummy_batch)

        assert (
            results == results_backload
        ), f"Results mismatch after backload. Results: {results}, Results backload: {results_backload}"
    finally:
        ray.shutdown()  # Clean up Ray resources after the test


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_type", "strategy"),
    [
        ("policy", "fsdp2"),
        ("critic", "fsdp2"),
        ("ref", "fsdp2"),
    ],
    ids=[
        "fsdp2_policy",
        "fsdp2_critic",
        "fsdp2_ref",
    ],
)
async def test_cpu_offload_correctness(ray_init_fixture, cfg, worker_type, strategy):
    """
    Test that the cpu_offload is working correctly for different backends.

    steps:
    1. Initialize actor group with the specified worker class.
    2. Make sure that the model is fully offloaded to cpu
    3. Run a forward pass and make sure that the memory is still close to 0
    """
    cfg.trainer.strategy = strategy
    # test that things work without any offloading setup by FSDP/FSDP2
    getattr(cfg.trainer, worker_type).fsdp_config.cpu_offload = True
    try:
        actor_group = init_worker_with_type(
            worker_type,
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )
        after_init = get_rank_0_memory(actor_group, "After init")

        # should be close to 0
        assert after_init < 1e8, f"Memory after offloading should be close to 0: instead {after_init} bytes"

        dummy_batch = make_dummy_tensorbatch()
        # Run forward pass
        ray.get(actor_group.async_run_ray_method("pass_through", "forward", dummy_batch))

        after_offload = get_rank_0_memory(actor_group, "After offload")

        # should still be relatively small
        assert after_offload < 4e8, f"Memory after forward pass should be < 400MB: instead {after_offload} bytes"

    finally:
        ray.shutdown()


@pytest.mark.parametrize(
    "strategy",
    [
        "fsdp",
        "fsdp2",
    ],
)
def test_offload_after_ckpt(ray_init_fixture, strategy):
    """
    Test ckpt+offload logic by:
    1. Creating model and doing one training step
    2. Saving checkpoint
    3. Offload parameters and optimizer
    4. Ensure that memory was freed
    """
    cfg = get_test_actor_config()
    ckpt_path = "$HOME/ckpts/test/"
    cfg.trainer.ckpt_path = ckpt_path
    cfg.trainer.export_path = ckpt_path
    cfg.trainer.strategy = strategy

    checkpoint_dir = None
    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )
        get_rank_0_memory(actor_group, "After init")

        # Create dummy training batch for training steps
        dp_size = actor_group.actor_infos[0].rank.dp_size
        dummy_batch_1 = make_dummy_training_batch(batch_size=dp_size)

        # Step 1: Do initial forward_backward + optim_step
        ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=dummy_batch_1))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))
        get_rank_0_memory(actor_group, "After training step 1")

        checkpoint_path = os.path.expandvars(os.path.join(cfg.trainer.ckpt_path, "global_step_1", "policy"))
        checkpoint_dir = os.path.expandvars(os.path.join(cfg.trainer.ckpt_path, "global_step_1"))  # Store for cleanup

        # Step 2: Save checkpoint
        ray.get(actor_group.async_run_ray_method("pass_through", "save_checkpoint", ckpt_dir=checkpoint_path))
        after_training = get_rank_0_memory(actor_group, "After ckpt")

        # Step 3:Offload model to CPU
        actor_group.offload_to_cpu()
        after_offload = get_rank_0_memory(actor_group, "After offload")

        # Step 4: Check that memory is offloaded
        offload_delta = after_training - after_offload
        assert offload_delta > 2.5 * 1024**3, f"Offload memory is {offload_delta} bytes, should be > 2.5GB"

    finally:
        # Clean up ray
        ray.shutdown()

        # Clean up checkpoint directory
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            print(f"Removing checkpoint directory: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)
