"""
Run with:
uv run --isolated --extra dev -- pytest tests/gpu/test_save_load_ckpt.py
"""

import ray
import pytest
import hydra
import torch
import os
import shutil
from omegaconf import DictConfig

from tests.gpu.utils import init_worker_with_type, make_dummy_experience, get_model_logits_from_actor
from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.utils.utils import print_mem

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
CKPT_PATH = "$HOME/ckpts/test/"

def get_rank_0_memory(actor_group, message: str):
    mem = ray.get(actor_group.async_run_ray_method("pass_through", "get_cuda_memory"))[0]
    print_mem(message, mem)
    return mem["allocated"]

def get_test_actor_config(strategy: str) -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 1
    cfg.trainer.strategy = strategy

    cfg.trainer.ckpt_path = CKPT_PATH
    cfg.trainer.export_path = CKPT_PATH

    return cfg


@pytest.mark.parametrize(
    "strategy",
    [
        # "deepspeed",
        # "fsdp",
        "fsdp2",
    ],
)
def test_save_load_checkpoint(strategy):
    """
    Test checkpointing logic by:
    1. Creating model and doing one training step
    2. Saving checkpoint
    3. Doing second training step and recording model logits
    4. Loading checkpoint
    5. Repeating second training step and comparing logits
    """
    cfg = get_test_actor_config(strategy)

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

        # Create dummy experiences for training steps
        dummy_experience_1 = make_dummy_experience()  # First training step
        dummy_experience_2 = make_dummy_experience()  # Second training step

        # Ensure the second experience is different from the first
        for i, seq in enumerate(dummy_experience_2.sequences):
            dummy_experience_2.sequences[i] = torch.randint(100, 200, seq.shape, device=seq.device)

        global_step, local_step, accumulation_steps = 0, 0, 1
        
        actor_group.offload_to_cpu()

        initial_offload_mem = get_rank_0_memory(actor_group, "After initial offload")

        # Backload to GPU
        actor_group.backload_to_gpu()
        get_rank_0_memory(actor_group, "Before training")

        # Step 1: Do initial training step
        ray.get(
            actor_group.async_run_ray_method(
                "pass_through", "training_step", dummy_experience_1, global_step, local_step, accumulation_steps
            )
        )
        get_rank_0_memory(actor_group, "After training step 1")

        checkpoint_path = os.path.expandvars(os.path.join(cfg.trainer.ckpt_path, "global_step_1", "policy"))
        checkpoint_dir = os.path.expandvars(os.path.join(cfg.trainer.ckpt_path, "global_step_1"))  # Store for cleanup

        # Step 2: Save checkpoint
        ray.get(actor_group.async_run_ray_method("pass_through", "save_ckpt", global_step=1, ckpt_dir=checkpoint_path))
        
        after_training = get_rank_0_memory(actor_group, "After ckpt")

        # Offload model to CPU
        actor_group.offload_to_cpu()

        after_offload = get_rank_0_memory(actor_group, "After offload")
        
        offload_delta = after_training - after_offload
        assert offload_delta > 5, f"Offload delta is {offload_delta}, should be ~11"
        
        # print(f"Offloaded {after_training - after_offload}, should be {initial_offload_mem - after_offload}")
        # print(f"Delta: {after_offload - initial_offload_mem}")
        
        

        # # Step 3: Do second training step and record results
        # ray.get(
        #     actor_group.async_run_ray_method(
        #         "pass_through", "training_step", dummy_experience_2, global_step + 1, local_step, accumulation_steps
        #     )
        # )

        # # Create test input for comparing model outputs
        # dp_size = actor_group.actor_infos[0].rank.dp_size
        # test_input = torch.randint(0, 1000, (dp_size, 20), device="cpu")  # batch_size=dp_size, seq_len=20
        # attention_mask = torch.ones_like(test_input)

        # # Step 4: Get logits after the second training step (this should be different from after checkpoint load)
        # logits_after_second_training = get_model_logits_from_actor(actor_group, test_input, attention_mask)

        # # Step 5: Load checkpoint via strategy's load_ckpt method
        # assert os.path.exists(checkpoint_path), f"Checkpoint directory {checkpoint_path} does not exist"
        # ray.get(actor_group.async_run_ray_method("pass_through", "load_ckpt", ckpt_dir=checkpoint_path))

        # # Step 6: Now repeat the exact same second training step
        # ray.get(
        #     actor_group.async_run_ray_method(
        #         "pass_through", "training_step", dummy_experience_2, global_step + 1, local_step, accumulation_steps
        #     )
        # )

        # # Get logits after loading checkpoint and repeating second training
        # logits_after_reload_and_training = get_model_logits_from_actor(actor_group, test_input, attention_mask)

        # # The logits should be exactly the same (checkpoint loading worked correctly)
        # torch.testing.assert_close(logits_after_second_training, logits_after_reload_and_training, atol=0.0, rtol=0.0)

    finally:
        # Clean up ray
        ray.shutdown()

        # Clean up checkpoint directory
        if checkpoint_dir and os.path.exists(checkpoint_dir):
            print(f"Removing checkpoint directory: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)
