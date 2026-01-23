"""
Fleet Task Training Entrypoint for SkyRL.

This entrypoint registers the FleetTaskEnv and runs GRPO training
on Fleet-hosted environments with checkpoint management.

Features:
- S3 checkpoint upload (if AWS credentials set)
- Aggressive local cleanup (keeps only 1 checkpoint to prevent disk exhaustion)

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

import hydra
import ray
from omegaconf import DictConfig
from skyrl_gym.envs import register
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray

logger = logging.getLogger(__name__)


class FleetPPOExp(BasePPOExp):
    """
    Fleet-specific PPO experiment with checkpoint management.

    Always wraps trainer to:
    - Clean up old checkpoints before saving (prevents disk exhaustion)
    - Upload to S3 if AWS credentials are set
    """

    def run(self):
        trainer = self._setup_trainer()

        # Always wrap trainer for checkpoint management
        # This handles both S3 upload (if credentials set) and local cleanup
        try:
            from integrations.fleet.s3_checkpoints import wrap_trainer_with_s3_upload

            trainer = wrap_trainer_with_s3_upload(trainer, keep_local=False)
        except Exception as e:
            logger.warning(f"Failed to setup checkpoint management: {e}")

        # Start the training loop
        asyncio.run(trainer.train())


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

    # Run training with checkpoint management
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
