"""
Fleet Task Training Entrypoint for SkyRL.

Registers the FleetTaskEnv and runs GRPO training on Fleet-hosted environments
with S3 checkpoint management.

Usage:
    python -m integrations.fleet.entrypoints.main_fleet \
        environment.env_class=fleet_task \
        environment.skyrl_gym.fleet_task.tasks_file=/path/to/tasks.json \
        data.train_data=./data/fleet/train.parquet \
        data.val_data=./data/fleet/validation.parquet

Environment Variables for S3 Checkpoint Management:
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    AWS_REGION: AWS region (default: us-east-1)
    S3_CHECKPOINT_BUCKET: S3 bucket name (default: skyrl-checkpoints)
    RESUME_RUN_NAME: Run name to resume from (downloads checkpoint from S3)
"""

import asyncio
import logging
import os
from pathlib import Path

import hydra
import ray
from skyrl_gym.envs import register
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

logger = logging.getLogger(__name__)


class FleetPPOExp(BasePPOExp):
    """Fleet-specific PPO experiment with S3 checkpoint management."""

    def run(self):
        trainer = self._setup_trainer()

        # Download checkpoint from S3 if RESUME_RUN_NAME is set (cross-VM resume)
        resume_run_name = os.environ.get("RESUME_RUN_NAME", "")
        if resume_run_name:
            try:
                from integrations.fleet.s3_checkpoints import download_checkpoint_from_s3

                ckpt_path = trainer.cfg.trainer.ckpt_path
                model_path = getattr(trainer.cfg.trainer.policy.model, "path", "unknown-model")
                model_name = Path(model_path).name
                project_name = getattr(trainer.cfg.trainer, "project_name", "skyrl")
                download_checkpoint_from_s3(
                    ckpt_path=ckpt_path,
                    run_name=resume_run_name,
                    project_name=project_name,
                    model_name=model_name,
                )
            except Exception as e:
                logger.warning(f"Failed to download checkpoint from S3: {e}")

        # Wrap trainer for checkpoint management (cleanup + S3 upload)
        try:
            from integrations.fleet.s3_checkpoints import wrap_trainer_with_s3_upload

            trainer = wrap_trainer_with_s3_upload(trainer)
        except Exception as e:
            logger.warning(f"Failed to setup checkpoint management: {e}")

        asyncio.run(trainer.train())


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    """Ray remote function that registers Fleet environment and runs training."""
    register(
        id="fleet_task",
        entry_point="skyrl_gym.envs.fleet_task.env:FleetTaskEnv",
    )
    exp = FleetPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: SkyRLTrainConfig) -> None:
    """Main entry point for Fleet task training."""
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
