"""
S3 Checkpoint Management for SkyRL Training.

Provides checkpoint upload to S3, download from S3 for resume, and local cleanup.

Key behavior:
- Cleans up old local checkpoints BEFORE saving new one (prevents disk full)
- Uploads to S3 asynchronously (non-blocking, training continues)
- Downloads checkpoint from S3 before training for cross-VM resume
- Uploads eval results to S3 for persistence

Usage:
    from integrations.fleet.s3_checkpoints import (
        wrap_trainer_with_s3_upload,
        download_checkpoint_from_s3,
        upload_eval_results_to_s3,
    )

    # Download checkpoint before training (for resume on new VM)
    download_checkpoint_from_s3(ckpt_path, run_name)

    trainer = wrap_trainer_with_s3_upload(trainer, bucket="skyrl-checkpoints")
    upload_eval_results_to_s3(local_dir, run_name, global_step)

Environment Variables:
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    AWS_REGION: AWS region (default: us-east-1)
    S3_CHECKPOINT_BUCKET: S3 bucket for checkpoints (default: skyrl-checkpoints)
    S3_TRAJECTORY_BUCKET: S3 bucket for eval trajectories (default: skyrl-trajectories)
"""

import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class S3CheckpointUploader:
    """
    Uploads checkpoint directories to S3 asynchronously.

    Uses a background thread pool to avoid blocking training.
    Deletes local checkpoints after successful upload.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        region: str = "us-east-1",
        max_workers: int = 2,
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="s3-upload")
        self._pending: set = set()
        self._lock = threading.Lock()

    def _upload_sync(self, local_dir: str) -> bool:
        """Synchronous upload that runs in thread pool."""
        try:
            import boto3
            from botocore.config import Config
            from boto3.s3.transfer import TransferConfig

            config = Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                connect_timeout=30,
                read_timeout=120,
            )

            s3 = boto3.client("s3", region_name=self.region, config=config)

            local_path = Path(local_dir)
            if not local_path.exists():
                logger.warning(f"Checkpoint directory does not exist: {local_dir}")
                return False

            checkpoint_name = local_path.name
            s3_prefix = f"{self.prefix}/{checkpoint_name}"

            transfer_config = TransferConfig(
                multipart_threshold=64 * 1024 * 1024,
                multipart_chunksize=64 * 1024 * 1024,
                max_concurrency=4,
                use_threads=True,
            )

            uploaded_files = 0
            total_size = 0

            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"{s3_prefix}/{relative_path}"
                    file_size = file_path.stat().st_size
                    total_size += file_size

                    logger.info(f"Uploading {file_path.name} ({file_size / 1e6:.1f} MB)")

                    s3.upload_file(str(file_path), self.bucket, s3_key, Config=transfer_config)
                    uploaded_files += 1

            logger.info(
                f"Uploaded {checkpoint_name}: {uploaded_files} files, {total_size / 1e9:.2f} GB to s3://{self.bucket}/{s3_prefix}/"
            )

            # Delete local after successful upload to free disk space
            logger.info(f"Deleting local checkpoint after S3 upload: {local_dir}")
            shutil.rmtree(local_dir)

            return True

        except Exception as e:
            logger.error(f"S3 upload failed for {local_dir}: {e}")
            return False
        finally:
            with self._lock:
                self._pending.discard(local_dir)

    def upload_async(self, local_dir: str) -> None:
        """Queue checkpoint for async upload. Non-blocking."""
        with self._lock:
            if local_dir in self._pending:
                return
            self._pending.add(local_dir)

        logger.info(f"Queuing checkpoint for S3 upload: {local_dir}")
        self._executor.submit(self._upload_sync, local_dir)

    def wait_for_uploads(self, timeout: Optional[float] = None) -> None:
        """Wait for all pending uploads to complete."""
        self._executor.shutdown(wait=True)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-upload")


def cleanup_old_local_checkpoints(ckpt_path: str, keep_n: int = 2) -> None:
    """
    Delete old local checkpoints, keeping only the most recent N.

    Args:
        ckpt_path: Base checkpoint directory
        keep_n: Number of recent checkpoints to keep (default: 2 for safety)
    """
    ckpt_dir = Path(ckpt_path)
    if not ckpt_dir.exists():
        return

    checkpoint_dirs = sorted(
        [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("global_step_")],
        key=lambda x: int(x.name.split("_")[-1]),
        reverse=True,
    )

    for old_dir in checkpoint_dirs[keep_n:]:
        logger.info(f"Cleaning up old local checkpoint: {old_dir}")
        try:
            shutil.rmtree(old_dir)
        except Exception as e:
            logger.warning(f"Failed to delete {old_dir}: {e}")


def wrap_trainer_with_s3_upload(
    trainer,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    region: Optional[str] = None,
):
    """
    Wrap a SkyRL trainer to:
    1. Clean up old checkpoints BEFORE saving (prevents disk full)
    2. Upload to S3 asynchronously AFTER saving (if credentials set)
    3. Delete local checkpoint after successful S3 upload (frees disk)

    Args:
        trainer: SkyRL trainer instance
        bucket: S3 bucket (default: from S3_CHECKPOINT_BUCKET env var)
        prefix: S3 prefix (default: from trainer config)
        region: AWS region (default: from AWS_REGION env var)

    Returns:
        The trainer (modified in place)
    """
    bucket = bucket or os.environ.get("S3_CHECKPOINT_BUCKET", "skyrl-checkpoints")
    region = region or os.environ.get("AWS_REGION", "us-east-1")

    # Build prefix from trainer config
    if prefix is None:
        run_name = getattr(trainer.cfg.trainer, "run_name", None)
        project_name = getattr(trainer.cfg.trainer, "project_name", "skyrl")
        model_path = getattr(trainer.cfg.trainer.policy.model, "path", "unknown-model")
        model_name = Path(model_path).name
        prefix = f"{project_name}/{model_name}/{run_name}" if run_name else f"{project_name}/{model_name}"

    # Check AWS credentials
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    s3_enabled = bool(aws_key and aws_secret)

    if s3_enabled:
        logger.info(f"S3 checkpoint upload ENABLED: s3://{bucket}/{prefix}/")
        uploader = S3CheckpointUploader(bucket=bucket, prefix=prefix, region=region)
    else:
        logger.warning(
            "AWS credentials not found. S3 upload DISABLED. "
            "Using aggressive local cleanup (keeping only 2 checkpoints). "
            "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to enable S3."
        )
        uploader = None

    original_save_checkpoints = trainer.save_checkpoints
    ckpt_path = trainer.cfg.trainer.ckpt_path

    def save_checkpoints_with_cleanup():
        """Wrapped save_checkpoints with pre-save cleanup and async S3 upload."""
        # CRITICAL: Clean up old checkpoints BEFORE saving to free disk space
        # With S3: keep only 1 (we have S3 backup), allows room for new checkpoint
        # Without S3: keep 2 for safety
        keep_n = 1 if s3_enabled else 2
        cleanup_old_local_checkpoints(ckpt_path, keep_n=keep_n)

        # Now save the new checkpoint (disk has space)
        original_save_checkpoints()

        # Queue async S3 upload (non-blocking)
        if s3_enabled and uploader:
            global_step = trainer.global_step
            checkpoint_dir = os.path.join(ckpt_path, f"global_step_{global_step}")
            if os.path.exists(checkpoint_dir):
                uploader.upload_async(checkpoint_dir)

    trainer.save_checkpoints = save_checkpoints_with_cleanup
    trainer._s3_uploader = uploader

    return trainer


def download_checkpoint_from_s3(
    ckpt_path: str,
    run_name: str,
    bucket: Optional[str] = None,
    region: Optional[str] = None,
    project_name: str = "fleet-task-grpo",
    model_name: str = "Qwen3-32B",
) -> bool:
    """
    Download the latest checkpoint from S3 for resume on a fresh VM.

    Looks for checkpoint directories under the S3 prefix matching the run_name,
    downloads the latest one, and writes latest_ckpt_global_step.txt.

    Args:
        ckpt_path: Local checkpoint directory (e.g., ~/ckpts/fleet_tool_use_32b)
        run_name: W&B run name used as S3 prefix (e.g., fleet_tool_use_32b_d7167c1c)
        bucket: S3 bucket (default: from S3_CHECKPOINT_BUCKET env var)
        region: AWS region (default: from AWS_REGION env var)
        project_name: Project name used in S3 prefix
        model_name: Model name used in S3 prefix

    Returns:
        True if checkpoint was downloaded, False otherwise
    """
    bucket = bucket or os.environ.get("S3_CHECKPOINT_BUCKET", "skyrl-checkpoints")
    region = region or os.environ.get("AWS_REGION", "us-east-1")

    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not (aws_key and aws_secret):
        logger.info("No AWS credentials, skipping S3 checkpoint download")
        return False

    # Check if local checkpoint already exists
    latest_file = os.path.join(ckpt_path, "latest_ckpt_global_step.txt")
    if os.path.exists(latest_file):
        with open(latest_file, "r") as f:
            step = f.read().strip()
        local_ckpt = os.path.join(ckpt_path, f"global_step_{step}")
        if os.path.exists(local_ckpt):
            logger.info(f"Local checkpoint already exists at step {step}, skipping S3 download")
            return False

    try:
        import boto3
        from botocore.config import Config

        config = Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=120,
        )
        s3 = boto3.client("s3", region_name=region, config=config)

        # S3 prefix matches what wrap_trainer_with_s3_upload builds
        s3_prefix = f"{project_name}/{model_name}/{run_name}/"

        # List all checkpoint directories in S3
        paginator = s3.get_paginator("list_objects_v2")
        checkpoint_steps = set()
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_prefix, Delimiter="/"):
            for prefix_obj in page.get("CommonPrefixes", []):
                dir_name = prefix_obj["Prefix"].rstrip("/").split("/")[-1]
                if dir_name.startswith("global_step_"):
                    try:
                        step = int(dir_name.split("_")[-1])
                        checkpoint_steps.add(step)
                    except ValueError:
                        pass

        if not checkpoint_steps:
            logger.info(f"No checkpoints found in s3://{bucket}/{s3_prefix}")
            return False

        latest_step = max(checkpoint_steps)
        s3_ckpt_prefix = f"{s3_prefix}global_step_{latest_step}/"
        local_ckpt_dir = os.path.join(ckpt_path, f"global_step_{latest_step}")

        logger.info(f"Downloading checkpoint step {latest_step} from s3://{bucket}/{s3_ckpt_prefix}")

        os.makedirs(local_ckpt_dir, exist_ok=True)

        downloaded_files = 0
        total_size = 0
        for page in paginator.paginate(Bucket=bucket, Prefix=s3_ckpt_prefix):
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                relative_path = s3_key[len(s3_ckpt_prefix) :]
                if not relative_path:
                    continue
                local_file = os.path.join(local_ckpt_dir, relative_path)
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                file_size = obj["Size"]
                total_size += file_size
                logger.info(f"Downloading {relative_path} ({file_size / 1e6:.1f} MB)")
                s3.download_file(bucket, s3_key, local_file)
                downloaded_files += 1

        # Write latest_ckpt_global_step.txt so SkyRL's resume_mode=latest can find it
        os.makedirs(ckpt_path, exist_ok=True)
        with open(latest_file, "w") as f:
            f.write(str(latest_step))

        logger.info(
            f"Downloaded checkpoint: {downloaded_files} files, {total_size / 1e9:.2f} GB "
            f"from s3://{bucket}/{s3_ckpt_prefix} to {local_ckpt_dir}"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to download checkpoint from S3: {e}")
        return False


def upload_eval_results_to_s3(
    local_dir: str,
    run_name: str,
    global_step: Optional[int] = None,
    bucket: Optional[str] = None,
    region: Optional[str] = None,
    delete_local: bool = False,
) -> bool:
    """
    Upload eval results directory to S3.

    Args:
        local_dir: Local directory containing eval JSONL files
        run_name: Run name for S3 prefix (e.g., "fleet_tool_use_abc123")
        global_step: Global step number (for organizing in S3)
        bucket: S3 bucket (default: from S3_TRAJECTORY_BUCKET env var)
        region: AWS region (default: from AWS_REGION env var)
        delete_local: If True, delete local files after upload

    Returns:
        True if upload succeeded, False otherwise
    """
    bucket = bucket or os.environ.get("S3_TRAJECTORY_BUCKET", "skyrl-trajectories")
    region = region or os.environ.get("AWS_REGION", "us-east-1")

    # Check AWS credentials
    aws_key = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not (aws_key and aws_secret):
        logger.warning("AWS credentials not found. Skipping S3 upload for eval results.")
        return False

    local_path = Path(local_dir)
    if not local_path.exists():
        logger.warning(f"Eval directory does not exist: {local_dir}")
        return False

    try:
        import boto3
        from botocore.config import Config

        config = Config(
            retries={"max_attempts": 3, "mode": "adaptive"},
            connect_timeout=30,
            read_timeout=60,
        )

        s3 = boto3.client("s3", region_name=region, config=config)

        # Build S3 prefix: evals/{run_name}/global_step_{N}/
        step_suffix = f"global_step_{global_step}" if global_step is not None else "eval_only"
        s3_prefix = f"evals/{run_name}/{step_suffix}"

        uploaded_files = 0
        total_size = 0

        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(local_path)
                s3_key = f"{s3_prefix}/{relative_path}"
                file_size = file_path.stat().st_size
                total_size += file_size

                s3.upload_file(str(file_path), bucket, s3_key)
                uploaded_files += 1

        logger.info(
            f"Uploaded eval results: {uploaded_files} files, {total_size / 1e6:.2f} MB "
            f"to s3://{bucket}/{s3_prefix}/"
        )

        if delete_local:
            shutil.rmtree(local_dir)
            logger.info(f"Deleted local eval directory: {local_dir}")

        return True

    except Exception as e:
        logger.error(f"S3 upload failed for eval results {local_dir}: {e}")
        return False
