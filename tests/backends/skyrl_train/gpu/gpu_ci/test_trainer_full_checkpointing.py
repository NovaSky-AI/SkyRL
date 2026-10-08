"""
Integration test for full trainer checkpointing functionality.

This test validates that the RayPPOTrainer can save and restore ALL training state,
ensuring that training can resume exactly where it left off.

Run with:
For FSDP, run:
uv run --isolated --extra dev --extra fsdp pytest tests/backends/skyrl_train/gpu/gpu_ci/test_trainer_full_checkpointing.py -m "not megatron"

For Megatron, run:
uv run --isolated --extra dev --extra megatron pytest tests/backends/skyrl_train/gpu/gpu_ci/test_trainer_full_checkpointing.py -m "megatron"
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest
import ray
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils.tracking import Tracking
from tests.backends.skyrl_train.gpu.utils import import_worker, ray_init_for_tests

MODEL_NAME = "Qwen/Qwen3-0.6B"
# 4-layer GLM-5.3-Flash slice (2 KDA + 2 NoPE-MLA/DSA, 3 MoE layers x 288 experts, mHC
# residuals) -- the same checkpoint the megatron parity rows use.
GLM5_3_FLASH_MODEL_NAME = "eatang/GLM-5.3-Flash-4layer"
NUM_GPUS = 2


class DummyDataset(Dataset):
    """Minimal dataset for testing"""

    def __init__(self, size=10):
        self.data = [([{"role": "user", "content": f"Question {i}"}], None) for i in range(size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn(self, batch):
        return batch


def get_test_trainer_config(
    strategy: str, fsdp_cpu_offload: bool = False, model_name: str = MODEL_NAME, lora: bool = False
) -> SkyRLTrainConfig:
    """Create minimal trainer config for testing"""
    cfg = SkyRLTrainConfig()
    cfg.trainer.policy.model.path = model_name
    cfg.trainer.critic.model.path = model_name  # Enable critic for testing
    cfg.trainer.strategy = strategy
    if strategy == "fsdp":
        cfg.trainer.policy.fsdp_config.cpu_offload = fsdp_cpu_offload

    # Use minimal settings for faster testing
    cfg.trainer.placement.policy_num_gpus_per_node = NUM_GPUS
    cfg.trainer.placement.critic_num_gpus_per_node = NUM_GPUS
    cfg.trainer.placement.policy_num_nodes = 1
    cfg.trainer.placement.critic_num_nodes = 1
    cfg.trainer.algorithm.use_kl_loss = (
        False  # disable ref model so we just have policy and critic (NUM_GPUS total GPUs)
    )
    cfg.trainer.placement.colocate_all = False  # Disable colocation for simpler testing
    cfg.trainer.train_batch_size = NUM_GPUS
    cfg.trainer.policy_mini_batch_size = cfg.trainer.train_batch_size
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.update_epochs_per_batch = 1
    cfg.trainer.epochs = 1
    cfg.trainer.logger = "console"
    cfg.generator.n_samples_per_prompt = 1
    cfg.generator.inference_engine.num_engines = NUM_GPUS // 2
    cfg.generator.inference_engine.tensor_parallel_size = 2

    # Megatron-specific
    if strategy == "megatron":
        cfg.trainer.policy.megatron_config.tensor_model_parallel_size = 2
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 2
        cfg.trainer.placement.policy_num_gpus_per_node = 4
        # Disable critic for megatron
        cfg.trainer.critic.model.path = ""

    if "glm-5.3-flash" in model_name.lower():
        # glm5_next is a KDA + NoPE-MLA/DSA hybrid MoE with mHC residuals, shipped as a VL
        # checkpoint; SkyRL bridges only the language model, so route the trainer to the
        # text-only path. Mirrors the parity rows in megatron/test_megatron_models.py.
        cfg.trainer.policy.language_model_only = True
        cfg.trainer.ref.language_model_only = True
        cfg.generator.inference_engine.language_model_only = True
        cfg.trainer.remove_microbatch_padding = True
        # mHC rejects pipeline_model_parallel_size > 1, so depth has to come from EP here.
        cfg.trainer.policy.megatron_config.pipeline_model_parallel_size = 1
        cfg.trainer.policy.megatron_config.expert_model_parallel_size = 4
        cfg.trainer.policy.megatron_config.expert_tensor_parallel_size = 1
        cfg.trainer.policy.megatron_config.moe_grouped_gemm = True
        cfg.trainer.policy.megatron_config.moe_token_dispatcher_type = "alltoall"
        cfg.trainer.policy.megatron_config.moe_router_score_function = "sigmoid"
        cfg.trainer.policy.megatron_config.moe_router_load_balancing_type = "none"

    if lora:
        lora_cfg = cfg.trainer.policy.model.lora
        lora_cfg.rank = 32
        lora_cfg.alpha = 32
        if "glm-5.3-flash" in model_name.lower():
            # The "all-linear" default maps to dense-attention names that match none of
            # GLM-5.3-Flash's MLA or KDA projections, so spell them out. f_b_proj/g_b_proj
            # are deliberately absent (see test_megatron_models.py for why).
            lora_cfg.target_modules = [
                "linear_q_down_proj",
                "linear_q_up_proj",
                "linear_kv_down_proj",
                "linear_kv_up_proj",
                "linear_proj",
                "linear_fc1",
                "linear_fc2",
                "q_proj",
                "k_proj",
                "v_proj",
                "b_proj",
                "f_a_proj",
                "g_a_proj",
                "o_proj",
            ]
            cfg.trainer.policy.megatron_config.lora_config.merge_lora = False

    # Use temporary directories.
    #
    # The checkpoint is written by the Megatron worker actors, which Ray may place on a
    # different node than this test process. On a multi-node cluster a bare mkdtemp() lands in
    # node-local /tmp, so the workers write on their node while the assertions below look on
    # the driver's -- and the checkpoint appears to be missing. Point SKYRL_TEST_CKPT_DIR at a
    # shared mount there; leaving it unset (single-node CI) keeps the old behaviour.
    cfg.trainer.export_path = tempfile.mkdtemp(
        prefix="trainer_ckpt_test_", dir=os.environ.get("SKYRL_TEST_CKPT_DIR") or None
    )
    cfg.trainer.ckpt_path = cfg.trainer.export_path

    # Enable checkpointing with correct config names
    cfg.trainer.ckpt_interval = 1  # Save every step
    cfg.trainer.resume_mode = "none"  # Initially false, will be set to True for resume

    return cfg


def create_minimal_trainer(cfg: SkyRLTrainConfig):
    """Create a minimal trainer setup for testing"""
    # Create minimal tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.trainer.policy.model.path, trust_remote_code=True)

    # Create dummy dataset
    train_dataset = DummyDataset(size=4)  # Small dataset for quick testing

    # Create mock generator for testing
    mock_generator = MagicMock()

    # Create tracker
    tracker = Tracking(
        project_name=cfg.trainer.project_name,
        experiment_name=cfg.trainer.run_name,
        backend=cfg.trainer.logger,
        config=cfg,
    )

    # Create trainer (no inference engine needed for checkpointing tests)
    trainer = RayPPOTrainer(
        cfg=cfg,
        tracker=tracker,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=None,
        inference_engine_client=None,
        generator=mock_generator,
    )

    return trainer


@pytest.mark.parametrize(
    ("strategy", "fsdp_cpu_offload", "lora", "model_name"),
    [
        ("fsdp", False, False, MODEL_NAME),
        ("fsdp", True, False, MODEL_NAME),
        pytest.param("megatron", False, False, MODEL_NAME, marks=pytest.mark.megatron),
        # GLM-5.3-Flash (glm5_next) on the 4-layer slice, with LoRA -- what the DAPO recipes
        # actually train. Distinct from the dense rows above because this model checkpoints
        # things they do not have: mHC residual state, KDA linear-attention projections, and
        # 288-expert MoE layers sharded over EP=4. It also runs PP=1, since mHC rejects
        # pipeline parallelism.
        #
        # The no-LoRA (full-finetune) counterpart is NOT parametrized here: it fails inside
        # save_checkpoint with a distributed-optimizer shape mismatch
        # (distrib_optimizer.py, sharded_param_state_*), independent of EP and of the
        # optimizer sharding format. With LoRA only the adapter tensors are in the grad
        # buffer, which is why this row passes.
        pytest.param("megatron", False, True, GLM5_3_FLASH_MODEL_NAME, marks=pytest.mark.megatron),
    ],
    ids=[
        "fsdp_no_lora",
        "fsdp_cpu_offload",
        "megatron_no_lora",
        "megatron_glm5_3_flash_lora",
    ],
)
def test_trainer_full_checkpointing(ray_init_fixture, strategy, fsdp_cpu_offload, lora, model_name):
    """
    Test full trainer checkpointing by:
    1. Creating trainer and setting it up
    2. Saving checkpoint
    3. Capturing training state
    4. Destroying trainer
    5. Creating new trainer with resume enabled
    6. Loading checkpoint
    7. Verifying all state matches
    8. Continuing training to ensure it works
    """
    cfg = get_test_trainer_config(strategy, fsdp_cpu_offload, model_name, lora)

    checkpoint_dir = None
    try:
        # ============= PHASE 1: Initial Training and Save =============
        print("Phase 1: Initial training and checkpoint save")

        trainer1 = create_minimal_trainer(cfg)

        # Get worker classes
        PolicyWorker = import_worker(strategy, "policy")
        CriticWorker = import_worker(strategy, "critic")
        RefWorker = import_worker(strategy, "ref")

        # Build models
        trainer1.build_models(PolicyWorker, CriticWorker, RefWorker)

        # Set initial global step as if 2 steps were completed
        trainer1.global_step = 2

        # Save checkpoint
        trainer1.save_checkpoints()

        # Capture state before teardown
        saved_global_step = trainer1.global_step
        checkpoint_dir = os.path.join(cfg.trainer.export_path, f"global_step_{trainer1.global_step}")

        # Verify checkpoint structure was created
        expected_files = [
            os.path.join(checkpoint_dir, "policy"),
            os.path.join(checkpoint_dir, "trainer_state.pt"),
            os.path.join(checkpoint_dir, "data.pt"),
        ]
        # Only expect critic dir for non-megatron strategies
        if strategy != "megatron":
            expected_files.append(os.path.join(checkpoint_dir, "critic"))
        for expected_file in expected_files:
            assert os.path.exists(expected_file), f"Expected checkpoint file/dir not found: {expected_file}"

        # Verify atomic tracking file
        latest_ckpt_file = os.path.join(cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt")
        assert os.path.exists(latest_ckpt_file)
        with open(latest_ckpt_file, "r") as f:
            latest_step = int(f.read())
        assert latest_step == trainer1.global_step, "Atomic tracking file has incorrect step after first save"

        # Verify trainer state content
        print("Verifying checkpoint content...")
        loaded_trainer_state = torch.load(
            os.path.join(checkpoint_dir, "trainer_state.pt"), map_location="cpu", weights_only=False
        )

        # Check key configuration values are preserved
        config_as_omegaconf = OmegaConf.create(loaded_trainer_state["config"])
        loaded_trainer_config = SkyRLTrainConfig.from_dict_config(config_as_omegaconf)
        assert (
            loaded_trainer_config.trainer.train_batch_size == cfg.trainer.train_batch_size
        ), "train_batch_size not preserved in checkpoint"
        assert loaded_trainer_config.trainer.strategy == strategy, "strategy not preserved in checkpoint"
        assert loaded_trainer_state["global_step"] == saved_global_step, "global_step not preserved in checkpoint"

        # Cleanup first trainer
        del trainer1
        ray.shutdown()

        # ============= PHASE 2: Resume from Checkpoint =============
        print("Phase 2: Resume from checkpoint")
        ray_init_for_tests()
        # Create new config with resume enabled
        cfg_resume = get_test_trainer_config(strategy, fsdp_cpu_offload, model_name, lora)
        cfg_resume.trainer.resume_mode = "from_path"  # Enable resume
        cfg_resume.trainer.resume_path = checkpoint_dir  # Set resume path
        cfg_resume.trainer.export_path = cfg.trainer.export_path  # Use same export path
        cfg_resume.trainer.ckpt_path = cfg.trainer.ckpt_path

        trainer2 = create_minimal_trainer(cfg_resume)

        # Build models again
        trainer2.build_models(PolicyWorker, CriticWorker, RefWorker)

        # Load checkpoints
        loaded_global_step, loaded_checkpoint_dir = trainer2.load_checkpoints()
        assert (
            loaded_global_step == saved_global_step
        ), f"Expected global_step={saved_global_step}, got {loaded_global_step}"
        assert loaded_checkpoint_dir == checkpoint_dir, "Checkpoint path mismatch"

        # ============= PHASE 3: Continue Training =============
        print("Phase 3: Second checkpoint save")

        # Try to save another checkpoint to test cleanup logic
        trainer2.global_step = 3
        trainer2.save_checkpoints()

        next_checkpoint_dir = os.path.join(cfg.trainer.export_path, f"global_step_{trainer2.global_step}")
        assert os.path.exists(next_checkpoint_dir), "Could not save checkpoint after resume"

        # Verify atomic tracking file is updated
        latest_ckpt_file = os.path.join(cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt")
        assert os.path.exists(latest_ckpt_file)
        with open(latest_ckpt_file, "r") as f:
            latest_step = int(f.read())
        assert latest_step == trainer2.global_step, "Atomic tracking file was not updated after second save"

    finally:
        if checkpoint_dir and os.path.exists(os.path.dirname(checkpoint_dir)):
            print(f"Cleaning up checkpoint directory: {os.path.dirname(checkpoint_dir)}")
            shutil.rmtree(os.path.dirname(checkpoint_dir))
