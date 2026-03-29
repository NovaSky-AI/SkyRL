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
from omegaconf import OmegaConf, open_dict
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.legacy import GENERATOR_TO_INFERENCE_ENGINE_FIELDS
from skyrl.train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

logger = logging.getLogger(__name__)


def _sync_legacy_generator_to_inference_engine(cfg):
    """Sync flat legacy generator.* CLI overrides into generator.inference_engine.*.

    The YAML has both flat legacy keys (generator.backend, etc.) and the structured
    generator.inference_engine section. CLI args override the flat keys, but
    validate_cfg reads from the structured section. This function copies the flat
    values into inference_engine so both stay in sync.
    """
    gen = cfg.generator
    with open_dict(gen):
        if not OmegaConf.is_missing(gen, "inference_engine"):
            ie = gen.inference_engine
            for old_field, new_field in GENERATOR_TO_INFERENCE_ENGINE_FIELDS.items():
                if OmegaConf.is_missing(gen, old_field):
                    continue
                try:
                    value = getattr(gen, old_field)
                except Exception:
                    continue
                target_field = new_field if new_field else old_field
                try:
                    setattr(ie, target_field, value)
                except Exception:
                    pass


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


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: SkyRLTrainConfig) -> None:
    """Main entry point for Fleet task training."""
    # Hydra loads the legacy YAML with flat generator.* keys (e.g. generator.backend)
    # that are also overridden by CLI args. The YAML also has a structured
    # generator.inference_engine section with defaults. Sync flat CLI overrides
    # into the structured section so validate_cfg sees the right values.
    _sync_legacy_generator_to_inference_engine(cfg)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
