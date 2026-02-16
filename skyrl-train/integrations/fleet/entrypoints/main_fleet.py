"""
Fleet Task Training Entrypoint for SkyRL.

This entrypoint registers the FleetTaskEnv and runs GRPO training
on Fleet-hosted environments with checkpoint management.

Features:
- S3 checkpoint upload (if AWS credentials set)
- S3 checkpoint download for cross-VM resume
- Local cleanup to prevent disk exhaustion (keeps latest checkpoint for resume)

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
from omegaconf import DictConfig
from skyrl_gym.envs import register
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray
from skyrl_train.utils.tracking import Tracking

logger = logging.getLogger(__name__)


class FleetPPOExp(BasePPOExp):
    """
    Fleet-specific PPO experiment with checkpoint management.

    Always wraps trainer to:
    - Download checkpoint from S3 if RESUME_RUN_NAME is set (cross-VM resume)
    - Clean up old checkpoints before saving (prevents disk exhaustion)
    - Upload to S3 if AWS credentials are set
    - Keep local checkpoints for same-VM resume
    - Resume W&B run when resuming from checkpoint
    """

    def get_tracker(self):
        """Override tracker to support W&B run resume when RESUME_RUN_NAME is set."""
        resume_run_name = os.environ.get("RESUME_RUN_NAME", "")
        if not resume_run_name:
            return super().get_tracker()

        # Look up the W&B run ID so we can resume logging to the same run
        wandb_id = self._lookup_wandb_run_id(self.cfg.trainer.project_name, resume_run_name)
        if wandb_id:
            # Override run_name in config to match the resumed run
            from omegaconf import OmegaConf

            OmegaConf.update(self.cfg, "trainer.run_name", resume_run_name)
            logger.info(f"Resuming W&B run: name={resume_run_name}, id={wandb_id}")
            return Tracking(
                project_name=self.cfg.trainer.project_name,
                experiment_name=resume_run_name,
                backends=self.cfg.trainer.logger,
                config=self.cfg,
                wandb_resume="allow",
                wandb_id=wandb_id,
            )

        logger.warning(f"Could not find W&B run '{resume_run_name}', creating new run")
        return super().get_tracker()

    @staticmethod
    def _lookup_wandb_run_id(project_name, run_name):
        """Look up W&B run ID from run name for resume."""
        try:
            import wandb

            api = wandb.Api()
            entity = os.environ.get("WANDB_ENTITY", api.default_entity)
            runs = api.runs(
                f"{entity}/{project_name}",
                filters={"display_name": run_name},
            )
            for run in runs:
                logger.info(f"Found W&B run: {run.name} (id={run.id}, state={run.state})")
                return run.id
            logger.warning(f"No W&B run found with name '{run_name}' in {entity}/{project_name}")
        except Exception as e:
            logger.warning(f"Failed to look up W&B run ID: {e}")
        return None

    def run(self):
        trainer = self._setup_trainer()

        # Download checkpoint from S3 if RESUME_RUN_NAME is set (for cross-VM resume)
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

        # Always wrap trainer for checkpoint management
        # keep_local=True so checkpoints persist for resume after crash/preemption
        try:
            from integrations.fleet.s3_checkpoints import wrap_trainer_with_s3_upload

            trainer = wrap_trainer_with_s3_upload(trainer)
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
