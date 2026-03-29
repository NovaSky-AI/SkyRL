"""
Task Generation Training Entrypoint for SkyRL.

Registers the TaskGenEnv and runs GRPO training for task generation
with S3 checkpoint management.

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
from omegaconf import OmegaConf
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.legacy import is_legacy_config, translate_legacy_config
from skyrl.train.entrypoints.main_base import BasePPOExp, config_dir
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray

logger = logging.getLogger(__name__)


class FleetPPOExp(BasePPOExp):
    """Fleet-specific PPO experiment with S3 checkpoint management."""

    def run(self):
        trainer = self._setup_trainer()

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

        try:
            from integrations.fleet.s3_checkpoints import wrap_trainer_with_s3_upload

            trainer = wrap_trainer_with_s3_upload(trainer)
        except Exception as e:
            logger.warning(f"Failed to setup checkpoint management: {e}")

        asyncio.run(trainer.train())


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig):
    """Ray remote function that registers TaskGenEnv and runs training."""
    # task_gen env is registered in skyrl_gym.envs.__init__ (after PR 3)
    exp = FleetPPOExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: SkyRLTrainConfig) -> None:
    """Main entry point for task generation training."""
    # Hydra loads the legacy YAML with flat generator.* keys.
    # Translate to the new generator.inference_engine.* structure
    # before validate_cfg accesses those fields.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    if is_legacy_config(cfg_dict):
        cfg_dict = translate_legacy_config(cfg_dict)
        cfg = OmegaConf.create(cfg_dict)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
