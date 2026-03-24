"""
Task Generation Training Entrypoint for SkyRL.

This entrypoint registers the TaskGenEnv and runs GRPO training
for task generation with checkpoint management.

The LLM generates (prompt, verifier) pairs for Fleet environments,
and reward comes from inner-loop rollouts measuring learnability.

Usage:
    python -m integrations.fleet.entrypoints.main_task_gen \
        environment.env_class=task_gen \
        data.train_data=./data/task_gen/train.parquet \
        data.val_data=./data/task_gen/validation.parquet
"""

import asyncio
import logging
import os
from pathlib import Path

import hydra
import ray
from omegaconf import DictConfig
from skyrl_train.entrypoints.main_base import BasePPOExp, config_dir, validate_cfg
from skyrl_train.utils import initialize_ray

logger = logging.getLogger(__name__)


class FleetPPOExp(BasePPOExp):
    """
    Fleet-specific PPO experiment with checkpoint management.

    Always wraps trainer to:
    - Download checkpoint from S3 if RESUME_RUN_NAME is set (cross-VM resume)
    - Clean up old checkpoints before saving (prevents disk exhaustion)
    - Upload to S3 if AWS credentials are set
    - Keep local checkpoints for same-VM resume
    """

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
    Ray remote function that registers the TaskGenEnv and runs training.

    This must be a Ray remote function because environment registration needs
    to happen in the worker processes, not the driver.
    """
    # task_gen env is already registered in skyrl_gym.envs.__init__
    # No explicit registration needed

    # Run training with checkpoint management
    exp = FleetPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for task generation training.

    Required configuration:
        environment.env_class: "task_gen"
        data.train_data: Path to training parquet file
        data.val_data: Path to validation parquet file
    """
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
