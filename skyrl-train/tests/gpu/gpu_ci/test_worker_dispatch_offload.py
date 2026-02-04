"""
Test WorkerDispatch automatic offload/onload with colocation policies.

Run with:
uv run --isolated --extra dev -- pytest tests/gpu/gpu_ci/test_worker_dispatch_offload.py -v

These tests validate that WorkerDispatch correctly manages GPU memory when
multiple models share the same GPU (colocate_all=True or colocate_policy_ref=True).
"""

import ray
import pytest
from ray.util.placement_group import placement_group

from tests.gpu.utils import make_dummy_training_batch, get_rank_0_memory
from skyrl_train.config import SkyRLConfig
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.utils import get_ray_pg_ready_with_timeout
from skyrl_train.workers.worker_dispatch import WorkerDispatch, GPUState
from tests.gpu.utils import import_worker
from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, RefWorker, CriticWorker
from skyrl_train.workers.worker import PPORayActorGroup

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_config() -> SkyRLConfig:
    cfg = SkyRLConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.use_sample_packing = False
    cfg.trainer.logger = "console"
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.ref.fsdp_config.cpu_offload = False

    validate_cfg(cfg)
    return cfg


def init_colocated_actor_group(
    worker_cls,
    shared_pg,
    cfg: SkyRLConfig,
) -> PPORayActorGroup:
    """Initialize an actor group that shares a placement group with others."""
    return PPORayActorGroup(
        cfg,
        num_nodes=1,
        num_gpus_per_node=1,
        ray_actor_type=worker_cls,
        pg=shared_pg,
        num_gpus_per_actor=0.4,  # Share GPU
        colocate_all=True,
        sequence_parallel_size=cfg.trainer.policy.sequence_parallel_size,
        record_memory=cfg.trainer.policy.record_memory,
    )


@pytest.mark.asyncio
async def test_colocate_all_only_one_model_on_gpu(ray_init_fixture):
    """
    Test that with colocate_all=True, only one model is on GPU at a time.

    Scenario:
    1. Initialize policy and ref on shared GPU
    2. Call dispatch.forward("ref", ...) - ref should be on GPU, policy offloaded
    3. Call dispatch.forward_backward("policy", ...) - policy on GPU, ref offloaded
    4. Verify memory drops when switching (indicates offload happened)
    """
    cfg = get_test_config()

    try:
        # Create shared placement group
        pg = placement_group([{"GPU": 1, "CPU": 2}], strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)

        # Initialize both actor groups on shared GPU
        policy_group = init_colocated_actor_group(PolicyWorker, pg, cfg)
        ref_group = init_colocated_actor_group(RefWorker, pg, cfg)

        # Init models - after init, models are on GPU
        ray.get(policy_group.async_init_model(cfg.trainer.policy.model.path))
        ray.get(ref_group.async_init_model(cfg.trainer.policy.model.path))

        # Create dispatch with colocate_all=True
        dispatch = WorkerDispatch(
            cfg,
            policy_actor_group=policy_group,
            ref_actor_group=ref_group,
        )

        # Mark both as on GPU after init
        dispatch._gpu_state["policy"] = GPUState(model_on_gpu=True, optimizer_on_gpu=True)
        dispatch._gpu_state["ref"] = GPUState(model_on_gpu=True, optimizer_on_gpu=False)

        # Manually offload both to start from clean state
        policy_group.offload_to_cpu()
        ref_group.offload_to_cpu()
        dispatch.mark_all_offloaded()

        dp_size = policy_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)

        # === Test 1: Load ref model ===
        dispatch.forward("ref", dummy_batch)

        # Verify state tracking
        assert dispatch._gpu_state["ref"].model_on_gpu, "ref should be marked on GPU"
        assert not dispatch._gpu_state["policy"].model_on_gpu, "policy should be marked offloaded"

        # Verify ref memory increased (measure from ref_group since it's a separate actor)
        ref_mem_after_load = get_rank_0_memory(ref_group, "After ref forward")
        assert ref_mem_after_load > 1e8, f"Ref model should use significant memory: {ref_mem_after_load}"

        # === Test 2: Switch to policy (should offload ref) ===
        dispatch.forward_backward("policy", dummy_batch)

        # Verify state tracking
        assert dispatch._gpu_state["policy"].model_on_gpu, "policy should be on GPU"
        assert dispatch._gpu_state["policy"].optimizer_on_gpu, "policy optimizer should be on GPU"
        assert not dispatch._gpu_state["ref"].model_on_gpu, "ref should be offloaded"

        # Verify policy is on GPU and ref was offloaded
        policy_mem = get_rank_0_memory(policy_group, "After policy forward_backward")
        ref_mem_after_offload = get_rank_0_memory(ref_group, "Ref after being offloaded")
        assert policy_mem > 1e8, f"Policy model should use significant memory: {policy_mem}"
        assert (
            ref_mem_after_offload < ref_mem_after_load
        ), f"Ref memory should decrease after offload: {ref_mem_after_offload} < {ref_mem_after_load}"

        # === Test 3: Switch back to ref (should offload policy) ===
        dispatch.forward("ref", dummy_batch)

        # Verify state tracking
        assert dispatch._gpu_state["ref"].model_on_gpu, "ref should be on GPU"
        assert not dispatch._gpu_state["policy"].model_on_gpu, "policy should be offloaded"
        assert not dispatch._gpu_state["policy"].optimizer_on_gpu, "policy optimizer should be offloaded"

        # Verify policy was offloaded
        policy_mem_after_offload = get_rank_0_memory(policy_group, "Policy after being offloaded")
        assert (
            policy_mem_after_offload < policy_mem
        ), f"Policy memory should decrease after offload: {policy_mem_after_offload} < {policy_mem}"

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_gpu_state_tracking_accuracy(ray_init_fixture):
    """
    Test that _gpu_state accurately reflects what's actually on GPU.

    This verifies the internal state tracking matches the actual offload/onload operations.
    """
    cfg = get_test_config()

    try:
        pg = placement_group([{"GPU": 1, "CPU": 2}], strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)

        policy_group = init_colocated_actor_group(PolicyWorker, pg, cfg)
        ref_group = init_colocated_actor_group(RefWorker, pg, cfg)

        ray.get(policy_group.async_init_model(cfg.trainer.policy.model.path))
        ray.get(ref_group.async_init_model(cfg.trainer.policy.model.path))

        dispatch = WorkerDispatch(
            cfg,
            policy_actor_group=policy_group,
            ref_actor_group=ref_group,
        )

        # Start from clean state
        policy_group.offload_to_cpu()
        ref_group.offload_to_cpu()
        dispatch.mark_all_offloaded()

        # Verify initial state
        assert dispatch._gpu_state["policy"] == GPUState(model_on_gpu=False, optimizer_on_gpu=False)
        assert dispatch._gpu_state["ref"] == GPUState(model_on_gpu=False, optimizer_on_gpu=False)

        # Load policy for training (needs model + optimizer)
        dp_size = policy_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)
        dispatch.forward_backward("policy", dummy_batch)

        assert dispatch._gpu_state["policy"] == GPUState(model_on_gpu=True, optimizer_on_gpu=True)
        assert dispatch._gpu_state["ref"] == GPUState(model_on_gpu=False, optimizer_on_gpu=False)

        # Load ref for inference (only needs model)
        dispatch.forward("ref", dummy_batch)

        assert dispatch._gpu_state["ref"] == GPUState(model_on_gpu=True, optimizer_on_gpu=False)
        assert dispatch._gpu_state["policy"] == GPUState(model_on_gpu=False, optimizer_on_gpu=False)

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_colocate_policy_critic_training_switch(ray_init_fixture):
    """
    Test switching between policy and critic training with colocate_all=True.

    This tests the common PPO training pattern where we alternate between
    training policy and critic on the same GPU.

    Scenario:
    1. Train policy (forward_backward + optim_step)
    2. Train critic (forward_backward + optim_step)
    3. Train policy again
    4. Verify correct offload/onload at each switch
    """
    cfg = get_test_config()

    try:
        pg = placement_group([{"GPU": 1, "CPU": 2}], strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)

        policy_group = init_colocated_actor_group(PolicyWorker, pg, cfg)
        critic_group = init_colocated_actor_group(CriticWorker, pg, cfg)

        ray.get(policy_group.async_init_model(cfg.trainer.policy.model.path))
        ray.get(critic_group.async_init_model(cfg.trainer.policy.model.path))

        dispatch = WorkerDispatch(
            cfg,
            policy_actor_group=policy_group,
            critic_actor_group=critic_group,
        )

        # Start from clean state
        policy_group.offload_to_cpu()
        critic_group.offload_to_cpu()
        dispatch.mark_all_offloaded()

        dp_size = policy_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)

        # === Step 1: Train policy ===
        dispatch.forward_backward("policy", dummy_batch)
        dispatch.optim_step("policy")

        assert dispatch._gpu_state["policy"].model_on_gpu
        assert dispatch._gpu_state["policy"].optimizer_on_gpu
        assert not dispatch._gpu_state["critic"].model_on_gpu
        assert not dispatch._gpu_state["critic"].optimizer_on_gpu

        policy_mem = get_rank_0_memory(policy_group, "After policy training")
        assert policy_mem > 1e8, f"Policy model should use significant memory: {policy_mem}"

        # === Step 2: Train critic (should offload policy) ===
        dispatch.forward_backward("critic", dummy_batch)
        dispatch.optim_step("critic")

        assert dispatch._gpu_state["critic"].model_on_gpu
        assert dispatch._gpu_state["critic"].optimizer_on_gpu
        assert not dispatch._gpu_state["policy"].model_on_gpu
        assert not dispatch._gpu_state["policy"].optimizer_on_gpu

        # Verify critic is loaded and policy was offloaded
        critic_mem = get_rank_0_memory(critic_group, "After critic training")
        policy_mem_after_offload = get_rank_0_memory(policy_group, "Policy after offload")
        assert critic_mem > 1e8, f"Critic model should use significant memory: {critic_mem}"
        assert (
            policy_mem_after_offload < policy_mem
        ), f"Policy memory should decrease after offload: {policy_mem_after_offload} < {policy_mem}"

        # === Step 3: Train policy again (should offload critic) ===
        dispatch.forward_backward("policy", dummy_batch)
        dispatch.optim_step("policy")

        assert dispatch._gpu_state["policy"].model_on_gpu
        assert dispatch._gpu_state["policy"].optimizer_on_gpu
        assert not dispatch._gpu_state["critic"].model_on_gpu
        assert not dispatch._gpu_state["critic"].optimizer_on_gpu

        # Verify policy is loaded again and critic was offloaded
        policy_mem_reloaded = get_rank_0_memory(policy_group, "Policy reloaded")
        critic_mem_after_offload = get_rank_0_memory(critic_group, "Critic after offload")
        assert policy_mem_reloaded > 1e8, f"Policy should be back on GPU: {policy_mem_reloaded}"
        assert (
            critic_mem_after_offload < critic_mem
        ), f"Critic memory should decrease after offload: {critic_mem_after_offload} < {critic_mem}"

    finally:
        ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("strategy"),
    ["fsdp2", pytest.param("megatron", marks=pytest.mark.megatron)],
    ids=["fsdp2", "megatron"],
)
async def test_dispatch_set_lr(ray_init_fixture, strategy):
    """
    Test that WorkerDispatch.set_lr updates the optimizer's learning rate.
    """
    cfg = get_test_config()
    cfg.trainer.strategy = strategy

    try:
        # Create placement group and policy actor
        pg = placement_group([{"GPU": 1, "CPU": 2}], strategy="PACK")
        get_ray_pg_ready_with_timeout(pg, timeout=30)

        policy_group = init_colocated_actor_group(import_worker(strategy, "policy"), pg, cfg)
        ray.get(policy_group.async_init_model(MODEL_NAME))

        dispatch = WorkerDispatch(cfg, policy_actor_group=policy_group)

        # Get initial learning rate
        initial_lrs = ray.get(policy_group.async_run_ray_method("pass_through", "get_lr"))
        initial_lr = initial_lrs[0]

        # Set a new learning rate via dispatch
        new_lr = 1e-5
        assert new_lr != initial_lr, "New LR should differ from initial for valid test"

        dispatch.set_lr("policy", new_lr)

        # Verify the learning rate was updated
        updated_lrs = ray.get(policy_group.async_run_ray_method("pass_through", "get_lr"))
        for updated_lr in updated_lrs:
            assert updated_lr == new_lr, f"Expected LR {new_lr}, got {updated_lr}"

    finally:
        ray.shutdown()
