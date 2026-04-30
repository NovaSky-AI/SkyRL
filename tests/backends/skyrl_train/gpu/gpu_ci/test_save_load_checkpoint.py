"""
For FSDP and FSDP2, run:
uv run --isolated --extra dev -- pytest tests/backends/skyrl_train/gpu/gpu_ci/test_save_load_checkpoint.py -m "not megatron"

For Megatron, run:
uv run --isolated --extra dev --extra mcore -- pytest tests/backends/skyrl_train/gpu/gpu_ci/test_save_load_checkpoint.py -m "megatron"

For Megatron cloud (S3) checkpoint test, run:
uv run --isolated --extra dev --extra mcore -- pytest tests/backends/skyrl_train/gpu/gpu_ci/test_save_load_checkpoint.py -k "test_save_load_checkpoint_cloud"
"""

import json
import os
import shutil
import uuid

import pytest
import ray
import torch
from transformers import AutoTokenizer

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.utils import print_mem, validate_cfg
from tests.backends.skyrl_train.gpu.utils import (
    get_model_logits_from_actor,
    init_worker_with_type,
    make_dummy_training_batch,
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
CKPT_PATH = "$HOME/ckpts/test/"
NUM_GPUS = 4


def run_one_training_step(
    actor_group,
    strategy,
    data=None,
    megatron_batch=None,
):
    """Run forward_backward + optim_step to perform one training step."""
    # Unified interface for all strategies (megatron, fsdp, fsdp2)
    batch = megatron_batch if strategy == "megatron" else data
    assert batch is not None, f"{strategy} requires a TrainingInputBatch for forward_backward"
    ray.get(actor_group.async_run_ray_method("mesh", "forward_backward", data=batch))
    ray.get(actor_group.async_run_ray_method("pass_through", "optim_step"))


def get_test_actor_config(strategy: str) -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.generator.inference_engine.tensor_parallel_size = NUM_GPUS
    cfg.trainer.strategy = strategy

    cfg.trainer.ckpt_path = CKPT_PATH
    cfg.trainer.export_path = CKPT_PATH
    cfg.trainer.logger = "console"

    validate_cfg(cfg)

    return cfg


@pytest.mark.parametrize(
    ("strategy", "lora", "fully_reshardable", "optimizer_cpu_offload"),
    [
        ("fsdp", False, False, False),
        ("fsdp2", False, False, False),
        pytest.param("megatron", False, False, False, marks=pytest.mark.megatron),
        pytest.param("megatron", False, False, True, marks=pytest.mark.megatron),
        pytest.param("megatron", True, False, False, marks=[pytest.mark.megatron, pytest.mark.lora]),
        pytest.param("megatron", False, True, False, marks=pytest.mark.megatron),
        pytest.param(
            "megatron",
            False,
            True,
            True,
            marks=[
                pytest.mark.megatron,
                pytest.mark.skip(
                    reason="fully_reshardable + cpu_offload has multiple upstream megatron-core bugs "
                    "(_set_main_param_and_optimizer_states KeyError on 'step', master_param key "
                    "mismatch with HybridDeviceOptimizer). dp_reshardable + cpu_offload works."
                ),
            ],
        ),
    ],
    ids=[
        "fsdp",
        "fsdp2",
        "megatron",
        "megatron_optimizer_cpu_offload",
        "megatron_lora",
        "megatron_fully_reshardable",
        "megatron_fully_reshardable_optimizer_cpu_offload",
    ],
)
def test_save_load_checkpoint(ray_init_fixture, strategy, lora, fully_reshardable, optimizer_cpu_offload):
    """
    Test checkpointing logic by:
    1. Creating model and doing one training step
    2. Saving checkpoint
    3. Doing second training step and recording model logits
    4. Loading checkpoint
    5. Repeating second training step and comparing logits
    """
    cfg = get_test_actor_config(strategy)
    if lora:
        from skyrl.train.config import SkyRLLoraConfig

        cfg.trainer.policy.model.lora = SkyRLLoraConfig(rank=32, alpha=32)
    if fully_reshardable:
        cfg.trainer.policy.megatron_config.dist_ckpt_optim_fully_reshardable = True
    if optimizer_cpu_offload:
        cfg.trainer.policy.megatron_config.optimizer_config_kwargs["optimizer_cpu_offload"] = True
        cfg.trainer.policy.megatron_config.optimizer_config_kwargs["optimizer_offload_fraction"] = 1

    checkpoint_dir = None
    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Create dummy training batches for training steps
        dp_size = actor_group.actor_infos[0].rank.dp_size
        dummy_batch_1 = make_dummy_training_batch(batch_size=dp_size)  # First training step
        dummy_batch_2 = make_dummy_training_batch(batch_size=dp_size)  # Second training step

        # Ensure the second batch is different from the first
        dummy_batch_2["sequences"] = torch.randint(100, 200, dummy_batch_2["sequences"].shape, device="cpu")

        # For Megatron, build training batches and reuse the second one pre/post checkpoint resume
        if "megatron" in strategy:
            from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_worker import (
                get_test_training_batch,
            )

            dp_size = actor_group.actor_infos[0].rank.dp_size
            train_batch_1 = get_test_training_batch(dp_size if dp_size % NUM_GPUS == 0 else NUM_GPUS)
            train_batch_2 = get_test_training_batch(dp_size if dp_size % NUM_GPUS == 0 else NUM_GPUS)
        else:
            train_batch_1 = None
            train_batch_2 = None

        # Step 1: Do initial training step
        run_one_training_step(
            actor_group,
            strategy,
            data=dummy_batch_1,
            megatron_batch=train_batch_1,
        )

        checkpoint_path = os.path.expandvars(os.path.join(cfg.trainer.ckpt_path, "global_step_1", "policy"))
        checkpoint_dir = os.path.expandvars(os.path.join(cfg.trainer.ckpt_path, "global_step_1"))  # Store for cleanup

        # Step 2: Save checkpoint
        ray.get(
            actor_group.async_run_ray_method(
                "pass_through", "save_checkpoint", ckpt_dir=checkpoint_path, tokenizer=tokenizer
            )
        )

        # Step 2.1: Make sure that offloading still works after saving checkpoint
        memory_after_saving = ray.get(actor_group.async_run_ray_method("pass_through", "get_cuda_memory"))[0]
        print_mem("memory after saving checkpoint", memory_after_saving)

        actor_group.offload_to_cpu()

        memory_after_offloading = ray.get(actor_group.async_run_ray_method("pass_through", "get_cuda_memory"))[0]
        print_mem("memory after offloading", memory_after_offloading)

        assert (
            memory_after_offloading["allocated"] < memory_after_saving["allocated"]
        ), f"Memory after offloading should be less than after saving: {memory_after_offloading} bytes < {memory_after_saving} bytes"
        actor_group.backload_to_gpu()

        # check that relevant files are saved
        huggingface_dir = os.path.join(checkpoint_path, "huggingface")
        expected_files = ["config.json", "generation_config.json", "tokenizer.json"]
        for file in expected_files:
            assert os.path.exists(
                os.path.join(huggingface_dir, file)
            ), f"File {file} not found in huggingface directory"
        if "fsdp" in strategy:
            fsdp_config_path = os.path.join(checkpoint_path, "fsdp_config.json")
            with open(fsdp_config_path, "r") as f:
                fsdp_config = json.load(f)
            assert fsdp_config["fsdp_strategy"] == strategy
            assert fsdp_config["world_size"] == NUM_GPUS

        # Step 3: Do second training step and record results
        run_one_training_step(
            actor_group,
            strategy,
            data=dummy_batch_2,
            megatron_batch=train_batch_2,
        )

        # Create test input for comparing model outputs
        dp_size = actor_group.actor_infos[0].rank.dp_size
        test_input = torch.randint(0, 1000, (dp_size, 20), device="cpu")  # batch_size=dp_size, seq_len=20
        attention_mask = torch.ones_like(test_input)

        # Step 4: Get logits after the second training step (this should be different from after checkpoint load)
        logits_after_second_training = get_model_logits_from_actor(actor_group, test_input, attention_mask)

        # Step 5: Load checkpoint via strategy's load_checkpoint method
        assert os.path.exists(checkpoint_path), f"Checkpoint directory {checkpoint_path} does not exist"
        ray.get(actor_group.async_run_ray_method("pass_through", "load_checkpoint", ckpt_dir=checkpoint_path))

        # Step 6: Now repeat the exact same second training step
        run_one_training_step(
            actor_group,
            strategy,
            data=dummy_batch_2,
            megatron_batch=train_batch_2,
        )

        # Get logits after loading checkpoint and repeating second training
        logits_after_reload_and_training = get_model_logits_from_actor(actor_group, test_input, attention_mask)

        # The logits should be exactly the same (checkpoint loading worked correctly)
        torch.testing.assert_close(logits_after_second_training, logits_after_reload_and_training, atol=0.0, rtol=0.0)

    finally:
        # Clean up checkpoint directory
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            print(f"Removing checkpoint directory: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)


@pytest.mark.megatron
def test_save_load_checkpoint_cloud(ray_init_fixture):
    """
    Test Megatron checkpoint save/load with cloud (S3) storage.

    Each rank should download only its own shard during load (per-rank
    downloading) rather than the entire checkpoint directory.

    Steps:
    1. Create model and do one training step
    2. Save checkpoint to S3
    3. Do second training step and record model logits
    4. Load checkpoint from S3 (per-rank download)
    5. Repeat second training step and compare logits
    """
    from skyrl.backends.skyrl_train.utils.io import io as skyrl_io

    S3_BASE = os.environ.get("ANYSCALE_ARTIFACT_STORAGE", None)
    if not S3_BASE:
        pytest.skip("ANYSCALE_ARTIFACT_STORAGE environment variable is not set")
    s3_ckpt_root = f"{S3_BASE}/test_ckpt_cloud_{uuid.uuid4().hex[:8]}"
    checkpoint_path = f"{s3_ckpt_root}/global_step_1/policy"

    cfg = get_test_actor_config("megatron")

    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=NUM_GPUS,
            cfg=cfg,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        dp_size = actor_group.actor_infos[0].rank.dp_size

        from tests.backends.skyrl_train.gpu.gpu_ci.megatron.test_megatron_worker import (
            get_test_training_batch,
        )

        train_batch_1 = get_test_training_batch(dp_size if dp_size % NUM_GPUS == 0 else NUM_GPUS)
        train_batch_2 = get_test_training_batch(dp_size if dp_size % NUM_GPUS == 0 else NUM_GPUS)

        # Step 1: Initial training step
        run_one_training_step(actor_group, "megatron", megatron_batch=train_batch_1)

        # Step 2: Save checkpoint to S3
        ray.get(
            actor_group.async_run_ray_method(
                "pass_through", "save_checkpoint", ckpt_dir=checkpoint_path, tokenizer=tokenizer
            )
        )

        # Step 3: Second training step and record logits
        run_one_training_step(actor_group, "megatron", megatron_batch=train_batch_2)

        test_input = torch.randint(0, 1000, (dp_size, 20), device="cpu")
        attention_mask = torch.ones_like(test_input)
        logits_after_second_training = get_model_logits_from_actor(actor_group, test_input, attention_mask)

        # Step 4: Load checkpoint from S3 (uses per-rank shard downloading)
        ray.get(actor_group.async_run_ray_method("pass_through", "load_checkpoint", ckpt_dir=checkpoint_path))

        # Step 5: Repeat second training step and compare
        run_one_training_step(actor_group, "megatron", megatron_batch=train_batch_2)

        logits_after_reload_and_training = get_model_logits_from_actor(actor_group, test_input, attention_mask)

        torch.testing.assert_close(logits_after_second_training, logits_after_reload_and_training, atol=0.0, rtol=0.0)

    finally:
        try:
            skyrl_io.remove(s3_ckpt_root)
            print(f"Cleaned up S3 checkpoint: {s3_ckpt_root}")
        except Exception as e:
            print(f"Warning: Failed to clean up S3 checkpoint at {s3_ckpt_root}: {e}")
