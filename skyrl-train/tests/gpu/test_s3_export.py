"""
Test S3 export functionality for checkpoints

Run with:
uv run --isolated --extra dev -- pytest tests/gpu/test_s3_export.py -v -s

Prerequisites:
- AWS credentials configured (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
- S3 bucket accessible for testing
- boto3 installed
"""

import ray
import pytest
import hydra
import torch
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from omegaconf import DictConfig
from transformers import AutoTokenizer

from tests.gpu.utils import init_worker_with_type, make_dummy_experience, get_model_logits_from_actor
from skyrl_train.entrypoints.main_base import config_dir

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture
def test_s3_bucket():
    """Get S3 bucket for testing from environment variable"""
    bucket = os.environ.get("SKYRL_TEST_S3_BUCKET")
    if not bucket:
        pytest.skip("SKYRL_TEST_S3_BUCKET environment variable not set")
    return bucket


@pytest.fixture
def temp_ckpt_dir():
    """Create temporary checkpoint directory"""
    temp_dir = Path(tempfile.mkdtemp(prefix="skyrl_s3_test_"))
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def get_s3_export_config(strategy: str, ckpt_path: str, s3_bucket: str, s3_prefix: str) -> DictConfig:
    """Get config with S3 export enabled"""
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")

    cfg.trainer.policy.model.path = MODEL_NAME
    cfg.trainer.placement.policy_num_gpus_per_node = 1  # Single GPU for testing
    cfg.trainer.strategy = strategy
    cfg.trainer.ckpt_path = str(ckpt_path)

    # Enable S3 export
    cfg.trainer.s3_export.enabled = True
    cfg.trainer.s3_export.bucket = s3_bucket
    cfg.trainer.s3_export.prefix = s3_prefix

    # Ensure we have minimal generator config for max_seq_len calculation
    cfg.generator.max_input_length = 256
    cfg.generator.sampling_params.max_generate_length = 128

    # Manually set max_seq_len in algorithm config (normally done in validate_config)
    # Need to create a new config object since the original is in struct mode
    from omegaconf import OmegaConf

    algorithm_config = OmegaConf.create(cfg.trainer.algorithm)
    algorithm_config.max_seq_len = cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length
    cfg.trainer.algorithm = algorithm_config

    # Disable CPU offload for stable checkpoint loading
    if "fsdp" in strategy:
        cfg.trainer.policy.fsdp_config.cpu_offload = False
        cfg.trainer.ref.fsdp_config.cpu_offload = False
        cfg.trainer.critic.fsdp_config.cpu_offload = False

    return cfg


def verify_s3_export(bucket: str, prefix: str, global_step: int) -> bool:
    """Verify that checkpoint was exported to S3"""
    try:
        cmd = ["aws", "s3", "ls", f"s3://{bucket}/{prefix}/global_step_{global_step}/", "--recursive"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        if result.returncode != 0:
            print(f"S3 ls failed: {result.stderr}")
            return False

        # Check if we have files in the output
        output_lines = [line.strip() for line in result.stdout.split("\n") if line.strip()]
        print(f"S3 files found: {len(output_lines)} files")
        for line in output_lines[:5]:  # Show first 5 files
            print(f"  - {line}")

        return len(output_lines) > 0

    except Exception as e:
        print(f"Error verifying S3 export: {e}")
        return False


def download_from_s3(bucket: str, prefix: str, local_dir: Path, global_step: int) -> bool:
    """Download checkpoint from S3 to local directory"""
    try:
        s3_path = f"s3://{bucket}/{prefix}/global_step_{global_step}/"
        local_path = local_dir / f"global_step_{global_step}"

        cmd = ["aws", "s3", "sync", s3_path, str(local_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)

        if result.returncode != 0:
            print(f"S3 download failed with return code {result.returncode}: {result.stderr}")
            return False

        # Verify files were downloaded
        if not local_path.exists():
            print(f"Download directory not created: {local_path}")
            return False

        files = list(local_path.rglob("*"))
        print(f"Downloaded {len(files)} files to {local_path}")
        return len(files) > 0

    except Exception as e:
        print(f"Error downloading from S3: {e}")
        import traceback

        traceback.print_exc()
        return False


def cleanup_s3_objects(bucket: str, prefix: str):
    """Cleanup S3 objects after test"""
    try:
        cmd = ["aws", "s3", "rm", f"s3://{bucket}/{prefix}/", "--recursive"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0:
            print(f"Cleaned up S3 objects: s3://{bucket}/{prefix}/")
        else:
            print(f"Warning: Failed to cleanup S3 objects: {result.stderr}")
    except Exception as e:
        print(f"Warning: Error during S3 cleanup: {e}")


@pytest.mark.parametrize(
    "strategy",
    [
        "fsdp2",
    ],
)
def test_s3_export_and_resume(ray_init_fixture, test_s3_bucket, temp_ckpt_dir, strategy):
    """
    Test S3 export and resume functionality by:
    1. Creating model and doing training step
    2. Saving checkpoint (should trigger S3 export)
    3. Verifying S3 export worked
    4. Clearing local checkpoints
    5. Downloading from S3
    6. Loading checkpoint and verifying model state is correct
    """

    # Generate unique prefix for this test run
    import time

    s3_prefix = f"gpu-test/s3-export-{int(time.time())}"

    print("\n=== S3 Export Test ===")
    print(f"Strategy: {strategy}")
    print(f"S3 Bucket: {test_s3_bucket}")
    print(f"S3 Prefix: {s3_prefix}")
    print(f"Local checkpoint dir: {temp_ckpt_dir}")

    cfg = get_s3_export_config(strategy, temp_ckpt_dir, test_s3_bucket, s3_prefix)

    try:
        # Step 1: Initialize worker
        print("\n--- Step 1: Initialize Worker ---")
        actor_group = init_worker_with_type(
            "policy",
            shared_pg=None,
            colocate_all=False,
            num_gpus_per_node=cfg.trainer.placement.policy_num_gpus_per_node,
            cfg=cfg,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Step 2: Do training step
        print("\n--- Step 2: Training Step ---")
        dummy_experience = make_dummy_experience()
        global_step = 1

        ray.get(actor_group.async_run_ray_method("pass_through", "training_step", dummy_experience, global_step, 0, 1))

        # Step 3: Save checkpoint (should trigger S3 export)
        print("\n--- Step 3: Save Checkpoint with S3 Export ---")
        checkpoint_path = temp_ckpt_dir / f"global_step_{global_step}" / "policy"
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Import and manually call S3 export (simulating trainer behavior)
        from skyrl_train.utils.trainer_utils import export_checkpoint_to_s3

        # Save local checkpoint first
        ray.get(
            actor_group.async_run_ray_method(
                "pass_through", "save_ckpt", global_step=global_step, ckpt_dir=str(checkpoint_path), tokenizer=tokenizer
            )
        )

        # Manually trigger S3 export (this is what trainer does)
        print("Triggering S3 export...")
        export_checkpoint_to_s3(
            bucket=test_s3_bucket,
            prefix=s3_prefix,
            local_checkpoint_dir=str(temp_ckpt_dir / f"global_step_{global_step}"),
            global_step=global_step,
        )

        # Step 4: Verify S3 export worked
        print("\n--- Step 4: Verify S3 Export ---")
        s3_export_success = verify_s3_export(test_s3_bucket, s3_prefix, global_step)
        assert s3_export_success, "S3 export verification failed"
        print("✅ S3 export verified successfully")

        # Step 5: Clear local checkpoints and download from S3
        print("\n--- Step 5: Clear Local and Download from S3 ---")

        # Get model state before clearing (for comparison)
        test_input = torch.randint(0, 1000, (1, 20), device="cpu")
        attention_mask = torch.ones_like(test_input)
        original_logits = get_model_logits_from_actor(actor_group, test_input, attention_mask)

        # Clear local checkpoints
        if temp_ckpt_dir.exists():
            shutil.rmtree(temp_ckpt_dir)
        temp_ckpt_dir.mkdir(exist_ok=True)

        # Download from S3
        download_success = download_from_s3(test_s3_bucket, s3_prefix, temp_ckpt_dir, global_step)
        assert download_success, "S3 download failed"
        print("✅ S3 download successful")

        # Step 6: Load checkpoint and verify
        print("\n--- Step 6: Load Checkpoint and Verify ---")
        downloaded_checkpoint_path = temp_ckpt_dir / f"global_step_{global_step}" / "policy"
        assert downloaded_checkpoint_path.exists(), f"Downloaded checkpoint not found: {downloaded_checkpoint_path}"

        # Load checkpoint
        ray.get(actor_group.async_run_ray_method("pass_through", "load_ckpt", ckpt_dir=str(downloaded_checkpoint_path)))

        # Verify model state is the same
        reloaded_logits = get_model_logits_from_actor(actor_group, test_input, attention_mask)
        torch.testing.assert_close(original_logits, reloaded_logits, atol=1e-6, rtol=1e-6)
        print("✅ Model state verification passed")

        print("\nS3 export and resume test PASSED!")

    finally:
        # Cleanup S3 objects
        print("\n--- Cleanup ---")
        cleanup_s3_objects(test_s3_bucket, s3_prefix)


if __name__ == "__main__":
    # For manual testing
    import sys

    if len(sys.argv) < 2:
        print("Usage: python test_s3_export.py <s3-bucket-name>")
        sys.exit(1)

    bucket = sys.argv[1]
    os.environ["SKYRL_TEST_S3_BUCKET"] = bucket

    # Run the test manually
    pytest.main([__file__, "-v", "-s"])
