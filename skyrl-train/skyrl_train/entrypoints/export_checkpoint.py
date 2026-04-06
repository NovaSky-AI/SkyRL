"""
Export a SkyRL checkpoint to HuggingFace format without running training.

Usage (same as training script, but different entrypoint):
    python -m skyrl_train.entrypoints.export_checkpoint \
        --config-path /path/to/your/config \
        trainer.ckpt_path=/path/to/ckpt \
        trainer.export_path=/path/to/export \
        trainer.policy.model.path=Qwen/Qwen2.5-1.5B-Instruct \
        trainer.strategy=fsdp2

Or with resume_mode=from_path for a specific step:
    python -m skyrl_train.entrypoints.export_checkpoint \
        --config-path /path/to/your/config \
        trainer.ckpt_path=/path/to/ckpt \
        trainer.export_path=/path/to/export \
        trainer.resume_mode=from_path \
        trainer.resume_path=/path/to/ckpt/global_step_100
"""

import os

import ray
from omegaconf import DictConfig, OmegaConf
import hydra

from skyrl_train.entrypoints.main_base import BasePPOExp, initialize_ray, config_dir
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.tracking import Tracking


class ExportExp(BasePPOExp):
    """Minimal experiment class for checkpoint export — no training."""

    def __init__(self, cfg: DictConfig):
        # Skip dataset loading and inference engine placement group setup
        self.cfg = cfg
        self.tokenizer = self.get_tokenizer()
        self.colocate_pg = None  # No inference engines needed

    def get_train_dataset(self):
        return None  # Not needed for export

    def get_eval_dataset(self):
        return None  # Not needed for export

    def get_inference_client(self):
        return None  # Not needed for export

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return None  # Not needed for export

    def get_tracker(self):
        return Tracking(
            project_name=self.cfg.trainer.project_name,
            experiment_name=self.cfg.trainer.run_name,
            backends=[],  # No logging backends needed for export
            config=self.cfg,
        )

    def _setup_trainer(self) -> RayPPOTrainer:
        """Override to skip inference engine creation and force colocate_all=False."""
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        # Force colocate_all=False — we have no inference engines, so the
        # assertion in build_models() (num_policy_gpus == num_rollout_gpus) would fail.
        OmegaConf.update(self.cfg, "trainer.placement.colocate_all", False, merge=True)

        if self.cfg.trainer.strategy in ("fsdp", "fsdp2"):
            from skyrl_train.workers.fsdp.fsdp_worker import PolicyWorker, CriticWorker, RefWorker
        elif self.cfg.trainer.strategy == "megatron":
            from skyrl_train.workers.megatron.megatron_worker import PolicyWorker, CriticWorker, RefWorker
        else:
            raise ValueError(f"Unknown strategy: {self.cfg.trainer.strategy}")

        tracker = self.get_tracker()

        trainer = RayPPOTrainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=self.tokenizer,
            train_dataset=None,  # Safe: _build_train_dataloader_and_compute_training_steps guards on None
            eval_dataset=None,
            inference_engine_client=None,
            generator=None,
            colocate_pg=None,
        )
        trainer.build_models(PolicyWorker, CriticWorker, RefWorker)
        return trainer

    def run(self):
        trainer = self._setup_trainer()

        # Load checkpoint (reads latest_ckpt_global_step.txt or cfg.trainer.resume_path)
        global_step, ckpt_path = trainer.load_checkpoints()
        if global_step == 0:
            raise RuntimeError(
                "No checkpoint found. Check trainer.ckpt_path or set "
                "trainer.resume_mode=from_path and trainer.resume_path=<path>"
            )
        trainer.global_step = global_step
        print(f"Loaded checkpoint from step {global_step}: {ckpt_path}")

        # Export to HuggingFace format (merged safetensors, collected on rank 0)
        trainer.save_models()
        export_dir = os.path.join(self.cfg.trainer.export_path, f"global_step_{global_step}", "policy")
        print(f"Exported HF model to: {export_dir}")


@ray.remote(num_cpus=1)
def export_entrypoint(cfg: DictConfig):
    exp = ExportExp(cfg)
    exp.run()


@hydra.main(config_path=config_dir, config_name="ppo_base_config", version_base=None)
def main(cfg: DictConfig) -> None:
    initialize_ray(cfg)
    ray.get(export_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
