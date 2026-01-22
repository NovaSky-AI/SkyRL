"""
Fleet Task Training Entrypoint for SkyRL.

This entrypoint registers the FleetTaskEnv and runs GRPO training
on Fleet-hosted environments with optional S3 checkpoint upload.

Usage:
    python -m integrations.fleet.entrypoints.main_fleet \
        environment.env_class=fleet_task \
        environment.skyrl_gym.fleet_task.tasks_file=/path/to/tasks.json \
        data.train_data=./data/fleet/train.parquet \
        data.val_data=./data/fleet/validation.parquet

Environment Variables for S3 Checkpoint Upload:
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    AWS_REGION: AWS region (default: us-east-1)
    S3_CHECKPOINT_BUCKET: S3 bucket name (default: skyrl-checkpoints)
"""

import asyncio
import logging
import os
from pathlib import Path

import hydra
import ray
from omegaconf import DictConfig
from skyrl_gym.envs import register
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray

logger = logging.getLogger(__name__)


class FleetPPOExp(BasePPOExp):
    """
    Fleet-specific PPO experiment with S3 checkpoint upload support.

    Wraps the trainer to upload checkpoints to S3 after each save,
    preventing disk exhaustion on cloud instances.
    """

    def run(self):
        trainer = self._setup_trainer()

        # Wrap trainer with S3 upload if credentials are available
        if os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"):
            try:
                from integrations.fleet.s3_checkpoints import wrap_trainer_with_s3_upload

                bucket = os.environ.get("S3_CHECKPOINT_BUCKET", "skyrl-checkpoints")
                region = os.environ.get("AWS_REGION", "us-east-1")

                # Build prefix from trainer config
                # Structure: {project}/{model}/{run_name}/
                project_name = getattr(self.cfg.trainer, "project_name", "fleet-task-grpo")
                run_name = getattr(self.cfg.trainer, "run_name", "unknown")

                # Extract model name from path (e.g., "Qwen/Qwen2.5-1.5B-Instruct" -> "Qwen2.5-1.5B-Instruct")
                model_path = getattr(self.cfg.trainer.policy.model, "path", "unknown-model")
                model_name = Path(model_path).name

                prefix = f"{project_name}/{model_name}/{run_name}"

                trainer = wrap_trainer_with_s3_upload(
                    trainer,
                    bucket=bucket,
                    prefix=prefix,
                    region=region,
                    keep_local=False,  # Delete local after upload to save disk
                )
                logger.info(f"S3 checkpoint upload enabled: s3://{bucket}/{prefix}/")
            except Exception as e:
                logger.warning(f"Failed to enable S3 checkpoint upload: {e}")
        else:
            logger.info("AWS credentials not found, S3 checkpoint upload disabled")

        # Start the training loop
        asyncio.run(trainer.train())

        # Wait for any pending S3 uploads before exiting
        if hasattr(trainer, "_s3_uploader"):
            logger.info("Waiting for pending S3 uploads to complete...")
            trainer._s3_uploader.wait_for_uploads(timeout=300)
            logger.info("All S3 uploads completed")


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: DictConfig):
    """
    Ray remote function that registers the Fleet environment and runs training.

    This must be a Ray remote function because environment registration needs
    to happen in the worker processes, not the driver.
    """
    # Register the Fleet task environment
    register(
        id="fleet_task",
        entry_point="integrations.fleet.env:FleetTaskEnv",
    )

    # Run training with S3 checkpoint support
    exp = FleetPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for Fleet task training.

    Required configuration:
        environment.env_class: "fleet_task"
        environment.skyrl_gym.fleet_task.tasks_file: Path to tasks JSON file
        data.train_data: Path to training parquet file
        data.val_data: Path to validation parquet file

    Optional configuration:
        environment.skyrl_gym.fleet_task.api_key: Fleet API key (or use FLEET_API_KEY env var)
        generator.max_turns: Maximum turns per episode (default: 50)
    """
    # Validate config args
    validate_cfg(cfg)

    # Initialize Ray cluster
    initialize_ray(cfg)

    # Run training in Ray
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
