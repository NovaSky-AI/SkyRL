"""
Run with:
uv run --isolated --extra dev -- pytest tests/gpu/gpu_ci/test_training_step.py
"""

import ray
import pytest

from tests.gpu.utils import init_worker_with_type, make_dummy_training_batch
from skyrl_train.utils.utils import print_mem, validate_cfg
from skyrl_train.config import SkyRLConfig


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MOE_MODEL_NAME = "Qwen/Qwen3-30B-A3B"


def get_test_actor_config() -> SkyRLConfig:
    cfg = SkyRLConfig()
    cfg.trainer.placement.policy_num_gpus_per_node = 2
    cfg.trainer.logger = "console"
    cfg.generator.inference_engine_tensor_parallel_size = 2

    return cfg


@pytest.fixture
def cfg() -> SkyRLConfig:
    return get_test_actor_config()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("packed", "strategy", "model_name"),
    [
        (True, "fsdp", MODEL_NAME),
        (False, "fsdp", MODEL_NAME),
        (True, "fsdp2", MODEL_NAME),
        (False, "fsdp2", MODEL_NAME),
        # TODO (erictang000): Add test for MoE model for FSDP backend
        # right now this fails due to token routing issues
        # (True, "fsdp2", MOE_MODEL_NAME),
    ],
    ids=["packed-fsdp", "unpacked-fsdp", "packed-fsdp2", "unpacked-fsdp2"],
)
async def test_policy_forward_backward_and_optim_step(ray_init_fixture, cfg, packed, strategy, model_name):
    """
    Full test: initialize actor group, send dummy experience to forward_backward + optim_step, validate output.
    """
    cfg.trainer.use_sample_packing = packed
    cfg.trainer.strategy = strategy
    cfg.trainer.policy.model.path = model_name

    validate_cfg(cfg)

    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Create TrainingInputBatch - worker's forward_backward handles micro-batching internally
        dp_size = actor_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)

        results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=dummy_batch))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

        memory = ray.get(actor_group.async_run_ray_method("pass_through", "get_cuda_memory"))
        memory = memory[0]
        print_mem("memory after forward_backward + optim_step", memory)

        for result in results:
            assert isinstance(result, dict), "Result should be a dictionary of training stats"
            assert "policy_loss" in result
            assert "loss_metrics/clip_ratio" in result
            assert "policy_entropy" in result
            for k, v in result.items():
                assert isinstance(v, (int, float)), f"{k} should be an int or float"

    finally:
        ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("packed", "strategy"),
    [(True, "fsdp"), (False, "fsdp"), (True, "fsdp2"), (False, "fsdp2")],
    ids=[
        "packed-fsdp",
        "unpacked-fsdp",
        "packed-fsdp2",
        "unpacked-fsdp2",
    ],
)
async def test_critic_forward_backward_and_optim_step(ray_init_fixture, cfg, packed, strategy):
    """
    Full test: initialize critic actor group, send dummy experience to forward_backward + optim_step, validate output.
    """
    cfg.trainer.use_sample_packing = packed
    cfg.trainer.strategy = strategy
    validate_cfg(cfg)
    try:
        actor_group = init_worker_with_type(
            "critic",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Create TrainingInputBatch - worker's forward_backward handles micro-batching internally
        dp_size = actor_group.actor_infos[0].rank.dp_size
        dummy_batch = make_dummy_training_batch(batch_size=dp_size)

        results = ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=dummy_batch))
        ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))

        for result in results:
            assert isinstance(result, dict), "Result should be a dictionary of training stats"
            assert "critic_loss" in result
            assert "values_mean" in result
            for k, v in result.items():
                assert isinstance(v, float), f"{k} should be a float"

    finally:
        ray.shutdown()


@pytest.mark.asyncio
async def test_set_lr_updates_optimizer(ray_init_fixture, cfg):
    """
    Test that set_lr updates the optimizer's learning rate.
    """
    cfg.trainer.use_sample_packing = False
    cfg.trainer.strategy = "fsdp2"
    validate_cfg(cfg)

    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        # Get initial learning rate
        initial_lrs = ray.get(actor_group.async_run_ray_method("pass_through", "get_lr"))
        initial_lr = initial_lrs[0]

        # Set a new learning rate
        new_lr = 1e-5
        assert new_lr != initial_lr, "New LR should differ from initial for valid test"

        ray.get(actor_group.async_run_ray_method("pass_through", "set_lr", learning_rate=new_lr))

        # Verify the learning rate was updated
        updated_lrs = ray.get(actor_group.async_run_ray_method("pass_through", "get_lr"))
        for updated_lr in updated_lrs:
            assert updated_lr == new_lr, f"Expected LR {new_lr}, got {updated_lr}"

    finally:
        ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("strategy"),
    ["fsdp2", pytest.param("megatron", marks=pytest.mark.megatron)],
    ids=["fsdp2", "megatron"],
)
async def test_sft_forward_backward_with_cross_entropy(ray_init_fixture, cfg, strategy):
    """
    Test SFT path: forward_backward with loss_fn="cross_entropy" returns loss_fn_outputs.
    Uses DP=2 to verify each rank returns outputs for its data chunk.
    """
    cfg.trainer.use_sample_packing = False
    cfg.trainer.strategy = strategy
    if strategy == "megatron":
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
        cfg.trainer.placement.policy_num_gpus_per_node = 2
    validate_cfg(cfg)

    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )

        dp_size = actor_group.actor_infos[0].rank.dp_size
        batch_size = dp_size * 2  # Ensure multiple samples per DP rank
        samples_per_rank = batch_size // dp_size
        num_actions = 4
        dummy_batch = make_dummy_training_batch(batch_size=batch_size, num_actions=num_actions)

        # Call forward_backward with loss_fn="cross_entropy"
        results = ray.get(
            actor_group.async_run_ray_method("mesh", "forward_backward", data=dummy_batch, loss_fn="cross_entropy")
        )

        # Each DP rank returns its chunk's results
        all_loss_fn_outputs = []
        for result in results:
            assert isinstance(result, dict)
            assert "loss" in result
            assert "loss_fn_outputs" in result, "SFT path should return loss_fn_outputs"

            loss_fn_outputs = result["loss_fn_outputs"]
            assert isinstance(loss_fn_outputs, list)
            assert len(loss_fn_outputs) == samples_per_rank, f"Expected {samples_per_rank} outputs per rank"
            all_loss_fn_outputs.extend(loss_fn_outputs)

        # Verify total outputs match batch size
        assert len(all_loss_fn_outputs) == batch_size, f"Expected {batch_size} total outputs"

        # Verify structure of each output
        for i, output in enumerate(all_loss_fn_outputs):
            assert "logprobs" in output, f"Output {i} missing logprobs"
            assert "elementwise_loss" in output, f"Output {i} missing elementwise_loss"
            # Verify trimmed to valid length (loss_mask is all 1s, so should equal num_actions)
            assert len(output["logprobs"]) == num_actions

    finally:
        ray.shutdown()
