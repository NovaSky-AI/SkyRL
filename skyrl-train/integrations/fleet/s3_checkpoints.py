"""
S3 Checkpoint Uploader for SkyRL Training.

Provides async upload of checkpoints to S3 with local cleanup to prevent disk exhaustion.

Usage:
    from integrations.fleet.s3_checkpoints import S3CheckpointUploader, wrap_trainer_with_s3_upload

    # Option 1: Wrap trainer to auto-upload after each checkpoint
    trainer = wrap_trainer_with_s3_upload(trainer, bucket="skyrl-checkpoints", prefix="run-name")

    # Option 2: Manual upload
    uploader = S3CheckpointUploader(bucket="skyrl-checkpoints", prefix="run-name")
    await uploader.upload_checkpoint("/path/to/checkpoint")

Environment Variables:
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    AWS_REGION: AWS region (default: us-east-1)
    S3_CHECKPOINT_BUCKET: S3 bucket name (default: skyrl-checkpoints)
"""

import asyncio
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)


class S3CheckpointUploader:
    """
    Uploads checkpoint directories to S3 asynchronously.

    Uses a background thread pool to avoid blocking training.
    Cleans up local checkpoints after successful upload to prevent disk exhaustion.
    """

    def __init__(
        self,
        bucket: str,
        prefix: str,
        region: str = "us-east-1",
        keep_local: bool = False,
        max_workers: int = 2,
    ):
        """
        Initialize S3 checkpoint uploader.

        Args:
            bucket: S3 bucket name
            prefix: S3 key prefix (e.g., "fleet-training/run-2024-01-22")
            region: AWS region
            keep_local: If True, don't delete local checkpoints after upload
            max_workers: Number of concurrent upload threads
        """
        self.bucket = bucket
        self.prefix = prefix
        self.region = region
        self.keep_local = keep_local
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="s3-upload")
        self._pending_uploads: set = set()
        self._lock = threading.Lock()

    def _upload_sync(self, local_dir: str) -> bool:
        """
        Synchronous upload function that runs in thread pool.

        Returns True on success, False on failure.
        """
        try:
            import boto3
            from botocore.config import Config

            config = Config(
                retries={"max_attempts": 3, "mode": "adaptive"},
                connect_timeout=30,
                read_timeout=120,
            )

            s3 = boto3.client(
                "s3",
                region_name=self.region,
                config=config,
            )

            local_path = Path(local_dir)
            if not local_path.exists():
                logger.warning(f"Checkpoint directory does not exist: {local_dir}")
                return False

            # Get relative path from parent (e.g., global_step_10)
            checkpoint_name = local_path.name
            s3_prefix = f"{self.prefix}/{checkpoint_name}"

            uploaded_files = 0
            total_size = 0

            for file_path in local_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(local_path)
                    s3_key = f"{s3_prefix}/{relative_path}"

                    file_size = file_path.stat().st_size
                    total_size += file_size

                    logger.info(f"Uploading {file_path} to s3://{self.bucket}/{s3_key} ({file_size / 1e6:.1f} MB)")

                    # Use multipart upload for large files
                    from boto3.s3.transfer import TransferConfig

                    transfer_config = TransferConfig(
                        multipart_threshold=64 * 1024 * 1024,  # 64MB
                        multipart_chunksize=64 * 1024 * 1024,
                        max_concurrency=4,
                        use_threads=True,
                    )

                    s3.upload_file(
                        str(file_path),
                        self.bucket,
                        s3_key,
                        Config=transfer_config,
                    )
                    uploaded_files += 1

            logger.info(
                f"Successfully uploaded checkpoint {checkpoint_name}: "
                f"{uploaded_files} files, {total_size / 1e9:.2f} GB total"
            )

            # Clean up local checkpoint after successful upload
            if not self.keep_local:
                logger.info(f"Cleaning up local checkpoint: {local_dir}")
                shutil.rmtree(local_dir)

            return True

        except Exception as e:
            logger.error(f"Failed to upload checkpoint {local_dir}: {e}")
            return False
        finally:
            with self._lock:
                self._pending_uploads.discard(local_dir)

    def upload_checkpoint(self, local_dir: str) -> None:
        """
        Queue a checkpoint directory for async upload.

        Non-blocking - returns immediately while upload happens in background.
        """
        with self._lock:
            if local_dir in self._pending_uploads:
                logger.warning(f"Checkpoint already queued for upload: {local_dir}")
                return
            self._pending_uploads.add(local_dir)

        logger.info(f"Queuing checkpoint for S3 upload: {local_dir}")
        self._executor.submit(self._upload_sync, local_dir)

    def wait_for_uploads(self, timeout: Optional[float] = None) -> None:
        """
        Wait for all pending uploads to complete.

        Args:
            timeout: Maximum seconds to wait (None = wait forever)
        """
        self._executor.shutdown(wait=True)
        # Recreate executor for future uploads
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="s3-upload")

    @property
    def pending_count(self) -> int:
        """Number of uploads currently in progress or queued."""
        with self._lock:
            return len(self._pending_uploads)


def wrap_trainer_with_s3_upload(
    trainer,
    bucket: Optional[str] = None,
    prefix: Optional[str] = None,
    region: Optional[str] = None,
    keep_local: bool = False,
):
    """
    Wrap a SkyRL trainer to upload checkpoints to S3 after each save.

    This monkey-patches the trainer's save_checkpoints method to add S3 upload.

    Args:
        trainer: SkyRL trainer instance
        bucket: S3 bucket (default: from S3_CHECKPOINT_BUCKET env var)
        prefix: S3 prefix (default: from trainer config run_name)
        region: AWS region (default: from AWS_REGION env var or us-east-1)
        keep_local: If True, keep local checkpoints after upload

    Returns:
        The trainer (modified in place)
    """
    # Get config from environment if not provided
    bucket = bucket or os.environ.get("S3_CHECKPOINT_BUCKET", "skyrl-checkpoints")
    region = region or os.environ.get("AWS_REGION", "us-east-1")

    # Get prefix from trainer config if not provided
    if prefix is None:
        run_name = getattr(trainer.cfg.trainer, "run_name", None)
        project_name = getattr(trainer.cfg.trainer, "project_name", "skyrl")
        prefix = f"{project_name}/{run_name}" if run_name else project_name

    # Check if AWS credentials are available
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        logger.warning(
            "AWS credentials not found in environment. S3 checkpoint upload disabled. "
            "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to enable."
        )
        return trainer

    logger.info(f"Enabling S3 checkpoint upload: s3://{bucket}/{prefix}/")

    # Create uploader
    uploader = S3CheckpointUploader(
        bucket=bucket,
        prefix=prefix,
        region=region,
        keep_local=keep_local,
    )

    # Store original method
    original_save_checkpoints = trainer.save_checkpoints

    def save_checkpoints_with_s3():
        """Wrapped save_checkpoints that uploads to S3 after local save."""
        # Call original save
        original_save_checkpoints()

        # Queue S3 upload for the checkpoint we just saved
        global_step = trainer.global_step
        ckpt_path = trainer.cfg.trainer.ckpt_path
        checkpoint_dir = os.path.join(ckpt_path, f"global_step_{global_step}")

        if os.path.exists(checkpoint_dir):
            uploader.upload_checkpoint(checkpoint_dir)
        else:
            logger.warning(f"Checkpoint directory not found after save: {checkpoint_dir}")

    # Monkey-patch the method
    trainer.save_checkpoints = save_checkpoints_with_s3

    # Store uploader reference on trainer for cleanup
    trainer._s3_uploader = uploader

    return trainer


def create_s3_uploader_from_env() -> Optional[S3CheckpointUploader]:
    """
    Create S3 uploader from environment variables.

    Returns None if AWS credentials are not configured.

    Environment Variables:
        S3_CHECKPOINT_BUCKET: Bucket name (required)
        S3_CHECKPOINT_PREFIX: Key prefix (required)
        AWS_REGION: Region (default: us-east-1)
        AWS_ACCESS_KEY_ID: AWS access key
        AWS_SECRET_ACCESS_KEY: AWS secret key
    """
    bucket = os.environ.get("S3_CHECKPOINT_BUCKET")
    prefix = os.environ.get("S3_CHECKPOINT_PREFIX")

    if not bucket or not prefix:
        logger.info("S3_CHECKPOINT_BUCKET or S3_CHECKPOINT_PREFIX not set, S3 upload disabled")
        return None

    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        logger.warning("AWS credentials not found, S3 upload disabled")
        return None

    region = os.environ.get("AWS_REGION", "us-east-1")

    return S3CheckpointUploader(bucket=bucket, prefix=prefix, region=region)
