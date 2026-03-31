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
import sys
from pathlib import Path

import ray
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

logger = logging.getLogger(__name__)


def _strip_hydra_prefixes(args: list[str]) -> list[str]:
    """Strip Hydra ++ and + prefixes from CLI args.

    from_cli_overrides rejects +/++ prefixed args, but our run scripts use
    them for environment-specific config (e.g. ++environment.skyrl_gym.task_gen.*).
    Since these fields now exist in the dataclass, we can strip the prefix.
    """
    cleaned = []
    for arg in args:
        if arg.startswith("++"):
            cleaned.append(arg[2:])
        elif arg.startswith("+"):
            cleaned.append(arg[1:])
        else:
            cleaned.append(arg)
    return cleaned


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
    """Ray remote function that runs Fleet training."""
    # fleet_task env is auto-registered by skyrl_gym.envs.__init__
    exp = FleetPPOExp(cfg)
    exp.run()


def main() -> None:
    """Main entry point for Fleet task training."""
    # Strip ++/+ prefixes from CLI args (used for env-specific config keys
    # that now have proper dataclass fields)
    args = _strip_hydra_prefixes(sys.argv[1:])
    # Build typed dataclass config (handles legacy flat→nested translation)
    cfg = SkyRLTrainConfig.from_cli_overrides(args)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
