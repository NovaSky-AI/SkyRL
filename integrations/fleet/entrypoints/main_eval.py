"""
Fleet Task Eval-Only Entrypoint for SkyRL.

Resumes a Fleet GRPO checkpoint from S3 (FSDP shards on every node), runs a
single evaluation pass over the eval dataset, and uploads the dumped eval
results to S3. No training loop, no optimizer state.

Mirrors the resume + weight-sync path used by `main_fleet.py:FleetPPOExp.run()`
so the same FSDP checkpoints can be replayed against the same eval set on a
fresh cluster (e.g. for variance bars across seeds).

Usage:
    python -m integrations.fleet.entrypoints.main_eval \
        environment.env_class=fleet_task \
        environment.skyrl_gym.fleet_task.tasks_file=/path/to/tasks.json \
        data.val_data=['/path/to/validation.parquet'] \
        trainer.policy.model.path=Qwen/Qwen3.5-9B \
        trainer.run_name=my_eval_run \
        trainer.dump_eval_results=true

Environment Variables for S3 Checkpoint Management:
    AWS_ACCESS_KEY_ID: AWS access key
    AWS_SECRET_ACCESS_KEY: AWS secret key
    AWS_REGION: AWS region (default: us-east-1)
    S3_CHECKPOINT_BUCKET: S3 bucket for FSDP checkpoints (default: skyrl-checkpoints)
    S3_TRAJECTORY_BUCKET: S3 bucket for eval result uploads (default: skyrl-trajectories)
    RESUME_RUN_NAME: Run name to resume from. If unset, evaluates the base
        weights at trainer.policy.model.path with no FSDP load.
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

    Matches `main_fleet.py`. `from_cli_overrides` rejects +/++ prefixed args,
    but our run scripts use them for environment-specific config keys that
    now exist in the dataclass — so we can strip the prefix safely.
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


class FleetEvalExp(BasePPOExp):
    """Fleet eval-only experiment with optional S3 checkpoint resume.

    Reuses the trainer's FSDP weight loading and inference-engine weight sync,
    then calls `trainer.eval()` once. `trainer.eval()` already handles local
    eval dump and S3 upload when `trainer.dump_eval_results=true`, so this
    entrypoint just needs to wire up resume + run a single eval pass.
    """

    def get_train_dataset(self):
        """No train dataset is needed for eval-only runs."""
        return None

    def run(self):
        trainer = self._setup_trainer()
        assert trainer.eval_dataloader is not None, (
            "FleetEvalExp requires an eval dataset. Set `data.val_data` "
            "and `trainer.eval_interval > 0`."
        )

        # Optional S3 resume: download FSDP shards on this VM and broadcast
        # to the rest of the cluster. Mirrors FleetPPOExp.run().
        resume_run_name = os.environ.get("RESUME_RUN_NAME", "")
        if resume_run_name:
            try:
                from integrations.fleet.s3_checkpoints import (
                    broadcast_checkpoint_to_workers,
                    download_checkpoint_from_s3,
                )

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
                broadcast_checkpoint_to_workers(ckpt_path)
            except Exception as e:
                logger.warning(f"Failed to download checkpoint from S3: {e}")

        asyncio.run(self._run_eval(trainer))

    async def _run_eval(self, trainer):
        """Initialize weight sync, load policy weights, and run eval once."""
        trainer.init_weight_sync_state()

        # Load only the policy FSDP shards. We bypass `trainer.load_checkpoints()`
        # because it also restores `train_dataloader.state_dict()`, which is None
        # in eval-only mode. Optimizer / lr scheduler state are skipped too.
        self._load_policy_only(trainer)

        # Push fresh weights to the inference engine for evaluation.
        await trainer.dispatch.save_weights_for_sampler()

        # `trainer.eval()` runs the eval loop and uploads to S3 when
        # `dump_eval_results=true`. The S3 prefix uses `trainer.global_step`,
        # which `_load_policy_only` sets from the resumed checkpoint.
        eval_metrics = await trainer.eval()
        trainer.tracker.log(eval_metrics, step=trainer.global_step, commit=True)
        trainer.tracker.finish()
        logger.info(f"Eval-only metrics: {eval_metrics}")

    def _load_policy_only(self, trainer):
        """Load only the policy FSDP shards from a `global_step_<N>` directory.

        Resolves the checkpoint path the same way `trainer.load_checkpoints()`
        does (LATEST via `latest_ckpt_global_step.txt`, or FROM_PATH via
        `cfg.trainer.resume_path`), then calls `dispatch.load_checkpoint`
        with optimizer / scheduler state disabled. Sets `trainer.global_step`
        so downstream eval dumps and S3 uploads land under the correct step.

        TODO: This duplicates the path-resolution half of
        `RayPPOTrainer.load_checkpoints()`. The reason for the duplication is
        that `load_checkpoints()` unconditionally calls
        `self.train_dataloader.load_state_dict(...)`, which crashes when
        `train_dataloader is None` (eval-only). If trainer ever grows a
        `skip_dataloader_state` / `policy_only` flag, drop this helper and
        call `trainer.load_checkpoints(...)` directly.
        """
        from skyrl.backends.skyrl_train.utils.io import io
        from skyrl.train.utils.trainer_utils import (
            GLOBAL_STEP_PREFIX,
            ResumeMode,
            extract_step_from_path,
            validate_consistency_for_latest_checkpoint,
        )

        if trainer.resume_mode == ResumeMode.NONE:
            logger.info("resume_mode=none; evaluating base model weights")
            return

        if trainer.resume_mode == ResumeMode.LATEST:
            latest_file = os.path.join(
                trainer.cfg.trainer.ckpt_path, "latest_ckpt_global_step.txt"
            )
            if not io.exists(latest_file):
                logger.warning(
                    "resume_mode=latest but no checkpoint found at "
                    f"{trainer.cfg.trainer.ckpt_path}; using base weights"
                )
                return
            with io.open_file(latest_file, "r") as f:
                step = int(f.read().strip())
            ckpt_dir = os.path.join(
                trainer.cfg.trainer.ckpt_path, f"{GLOBAL_STEP_PREFIX}{step}"
            )
            validate_consistency_for_latest_checkpoint(
                trainer.cfg.trainer.ckpt_path,
                step,
                ckpt_dir,
                latest_file,
                trainer.cfg.trainer.ckpt_interval,
            )
        else:  # ResumeMode.FROM_PATH
            ckpt_dir = str(trainer.cfg.trainer.resume_path)

        if not io.exists(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint path not found: {ckpt_dir}")

        global_step = extract_step_from_path(Path(ckpt_dir))
        if global_step == -1:
            raise ValueError(f"Checkpoint path is not a valid global_step dir: {ckpt_dir}")
        trainer.global_step = global_step

        policy_ckpt_dir = os.path.join(ckpt_dir, "policy")
        logger.info(f"Loading policy checkpoint from {policy_ckpt_dir} (step {global_step})")
        trainer.dispatch.load_checkpoint(
            "policy",
            policy_ckpt_dir,
            load_optimizer_states=False,
            load_lr_scheduler_states=False,
        )
        logger.info("Successfully loaded policy checkpoint for eval")


@ray.remote(num_cpus=1)
def skyrl_eval_entrypoint(cfg: SkyRLTrainConfig):
    """Ray remote function that runs Fleet eval-only."""
    # fleet_task env is auto-registered by skyrl_gym.envs.__init__
    exp = FleetEvalExp(cfg)
    exp.run()


def main() -> None:
    """Main entry point for Fleet task eval-only."""
    args = _strip_hydra_prefixes(sys.argv[1:])
    cfg = SkyRLTrainConfig.from_cli_overrides(args)
    validate_cfg(cfg)
    initialize_ray(cfg)
    ray.get(skyrl_eval_entrypoint.remote(cfg))


if __name__ == "__main__":
    main()
