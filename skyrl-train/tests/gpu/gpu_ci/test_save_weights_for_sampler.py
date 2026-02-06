"""
Test save_weights_for_sampler() method

GPU Requirements: 2 GPUs

Run with:
uv run --isolated --extra dev --extra vllm pytest tests/gpu/gpu_ci/test_save_weights_for_sampler.py -v
"""

import asyncio

import pytest
import ray
from skyrl_train.config import SkyRLConfig
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.workers.worker_dispatch import WorkerDispatch

from tests.gpu.utils import (
    get_test_prompts,
    init_inference_engines,
    init_worker_with_type,
    make_dummy_training_batch,
    run_inference,
)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def get_test_config() -> SkyRLConfig:
    """Get base config with test-specific overrides."""
    cfg = SkyRLConfig()
    cfg.trainer.policy.model.path = MODEL
    cfg.trainer.critic.model.path = ""
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.async_engine = True
    cfg.generator.num_inference_engines = 1
    cfg.generator.run_engines_locally = True
    cfg.trainer.use_sample_packing = False
    cfg.trainer.logger = "console"

    validate_cfg(cfg)

    return cfg


@pytest.mark.parametrize(
    ("colocate_all", "strategy", "backend"),
    [
        pytest.param(False, "fsdp2", "vllm", marks=pytest.mark.vllm),
        pytest.param(True, "fsdp2", "vllm", marks=pytest.mark.vllm),
    ],
    ids=[
        "no_colocate_fsdp2_vllm",
        "colocate_fsdp2_vllm",
    ],
)
def test_save_weights_for_sampler_then_inference(ray_init_fixture, colocate_all, strategy, backend):
    """
    Test that save_weights_for_sampler() correctly syncs weights before sampling.

    This test validates the Tinker API pattern:
    1. Train: forward_backward + optim_step
    2. Sync: save_weights_for_sampler()
    3. Sample: inference engine generates with updated weights
    """
    try:
        cfg = get_test_config()
        cfg.trainer.placement.colocate_all = colocate_all
        cfg.trainer.strategy = strategy
        cfg.generator.backend = backend

        client, pg, router, server_group = init_inference_engines(
            model=MODEL,
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=cfg.trainer.placement.colocate_all,
            backend=backend,
            sleep_level=2,  # Full sleep since we explicitly sync weights
        )

        # Initialize policy worker
        policy_group = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=cfg.trainer.placement.colocate_all,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Create WorkerDispatch with handle to inference_engine_client
        dispatch = WorkerDispatch(
            cfg=cfg,
            policy_actor_group=policy_group,
            inference_engine_client=client,
        )

        # Initialize weight sync state
        dispatch.init_weight_sync_state(client)

        # If colocate_all, sleep inference engine to free GPU memory for training
        if colocate_all:
            asyncio.run(client.sleep())
            dispatch.mark_all_offloaded()

        # === Step 1: Do a training step ===
        dp_size = policy_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)

        # Training: forward_backward + optim_step
        dispatch.forward_backward("policy", dummy_batch)
        grad_norm = dispatch.optim_step("policy")
        assert grad_norm is not None, "optim_step should return gradient norm"

        # === Step 2: Call save_weights_for_sampler ===
        asyncio.run(dispatch.save_weights_for_sampler())

        # === Step 3: Sample using inference engine ===
        asyncio.run(client.reset_prefix_cache())
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL, num_samples=5), sampling_params))

        # Verify we got responses
        assert "responses" in outputs, "Inference should return responses"
        assert len(outputs["responses"]) == 5, f"Expected 5 responses, got {len(outputs['responses'])}"

        # Verify responses are non-empty
        for i, response in enumerate(outputs["responses"]):
            assert len(response) > 0, f"Response {i} should not be empty"

        print(f"Example output: {outputs['responses'][0][:3]}...")

    finally:
        ray.shutdown()


@pytest.mark.parametrize("backend", [pytest.param("vllm", marks=pytest.mark.vllm)])
def test_save_weights_for_sampler_multiple_training_steps(ray_init_fixture, backend):
    """
    Test that multiple training steps followed by one save_weights_for_sampler works correctly.
    """
    try:
        cfg = get_test_config()
        cfg.trainer.placement.colocate_all = False
        cfg.trainer.strategy = "fsdp2"
        cfg.generator.backend = backend

        # Initialize inference engine (uses 1 GPU)
        client, pg, router, server_group = init_inference_engines(
            model=MODEL,
            cfg=cfg,
            use_local=True,
            async_engine=cfg.generator.async_engine,
            tp_size=cfg.generator.inference_engine_tensor_parallel_size,
            colocate_all=False,
            backend=backend,
            sleep_level=2,
        )

        # Initialize policy worker (uses 1 GPU)
        policy_group = init_worker_with_type(
            "policy",
            shared_pg=pg,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Create WorkerDispatch
        dispatch = WorkerDispatch(
            cfg=cfg,
            policy_actor_group=policy_group,
            inference_engine_client=client,
        )

        # Initialize weight sync state
        dispatch.init_weight_sync_state(client)

        # Do multiple training steps WITHOUT syncing
        dp_size = policy_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)

        for _ in range(3):
            dispatch.forward_backward("policy", dummy_batch)
            dispatch.optim_step("policy")

        # Now sync once - should sync all accumulated changes
        asyncio.run(dispatch.save_weights_for_sampler())

        # Verify inference works
        asyncio.run(client.reset_prefix_cache())
        sampling_params = get_sampling_params_for_backend(cfg.generator.backend, cfg.generator.sampling_params)
        outputs = asyncio.run(run_inference(client, get_test_prompts(MODEL, num_samples=2), sampling_params))
        assert len(outputs["responses"]) == 2, "Should get 2 responses"

    finally:
        ray.shutdown()
