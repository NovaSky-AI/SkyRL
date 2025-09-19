"""
For FSDP and DeepSpeed, run:
uv run --isolated --extra dev --extra deepspeed -- pytest tests/gpu/gpu_ci/test_save_load_checkpoint.py -m "not megatron"

For Megatron, run:
uv run --isolated --extra dev --extra mcore -- pytest tests/gpu/gpu_ci/test_save_load_checkpoint.py -m "megatron"
"""

import ray
import pytest
import hydra
import torch
import os
import shutil
import json
from omegaconf import DictConfig
from transformers import AutoTokenizer

from tests.gpu.utils import init_worker_with_type, make_dummy_experience, get_model_logits_from_actor, validate_cfg
from skyrl_train.entrypoints.main_base import config_dir

MODEL_NAME = "Qwen/Qwen3-0.6B"
CKPT_PATH = "$HOME/ckpts/test/"
NUM_GPUS = 4


def run_one_training_step(
    actor_group,
    strategy,
    experience=None,
    global_step=None,
    local_step=None,
    accumulation_steps=None,
    megatron_batch=None,
):
    if strategy == "megatron":
        assert megatron_batch is not None, "Megatron requires a TrainingInputBatch for ppo_train"
        return ray.get(actor_group.async_run_ray_method("mesh", "ppo_train", megatron_batch))
    else:
        assert experience is not None, f"{strategy} requires an Experience for training_step"
        return ray.get(
            actor_group.async_run_ray_method(
                "pass_through", "training_step", experience, global_step, local_step, accumulation_steps
            )
        )


def get_test_actor_config(strategy: str) -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.strategy = strategy

    cfg.trainer.ckpt_path = CKPT_PATH
    cfg.trainer.export_path = CKPT_PATH
    cfg.trainer.logger = "console"

    validate_cfg(cfg)

    return cfg


@pytest.mark.parametrize(
    "strategy",
    [
        "deepspeed",
        "fsdp",
        "fsdp2",
        pytest.param("megatron", marks=pytest.mark.megatron),
    ],
)
def test_save_load_checkpoint(ray_init_fixture, strategy):
    """
    Test checkpointing logic by:
    1. Creating model and doing one training step
    2. Saving checkpoint
    3. Doing second training step and recording model logits
    4. Loading checkpoint
    5. Repeating second training step and comparing logits
    """
    cfg = get_test_actor_config(strategy)

    try:
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        checkpoint_dir = None
        # Create dummy experiences for training steps
        dummy_experience_1 = make_dummy_experience()  # First training step
        dummy_experience_2 = make_dummy_experience()  # Second training step

        # Ensure the second experience is different from the first
        for i, seq in enumerate(dummy_experience_2.sequences):
            dummy_experience_2.sequences[i] = torch.randint(100, 200, seq.shape, device=seq.device)

        # For Megatron, build training batches and reuse the second one pre/post checkpoint resume
        if "megatron" in strategy:
            from tests.gpu.test_megatron_worker import get_test_training_batch

            dp_size = actor_group.actor_infos[0].rank.dp_size
            train_batch_1 = get_test_training_batch(dp_size if dp_size % NUM_GPUS == 0 else NUM_GPUS)
            train_batch_2 = get_test_training_batch(dp_size if dp_size % NUM_GPUS == 0 else NUM_GPUS)
        else:
            train_batch_1 = None
            train_batch_2 = None

        global_step, local_step, accumulation_steps = 0, 0, 1

        # Step 1: Do initial training step
        run_one_training_step(
            actor_group,
            strategy,
            experience=dummy_experience_1,
            global_step=global_step,
            local_step=local_step,
            accumulation_steps=accumulation_steps,
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
            experience=dummy_experience_2,
            global_step=global_step + 1,
            local_step=local_step,
            accumulation_steps=accumulation_steps,
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
            experience=dummy_experience_2,
            global_step=global_step + 1,
            local_step=local_step,
            accumulation_steps=accumulation_steps,
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
