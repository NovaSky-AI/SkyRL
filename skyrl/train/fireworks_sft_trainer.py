"""Hosted Fireworks specialization of SkyRL's SFT trainer."""

from __future__ import annotations

import asyncio
import json
import os

from skyrl.backends.fireworks.runtime import FireworksRuntime, PromotableCheckpoint
from skyrl.backends.fireworks.sft import FireworksSFTDispatch
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.train.sft_trainer import SFTTrainer
from skyrl.utils.tok import get_tokenizer


class FireworksSFTTrainer(SFTTrainer):
    """Reuse the native SFT loop while Fireworks owns model computation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._fireworks_runtime: FireworksRuntime | None = None
        self._ray_gpu_monitor = None
        self._num_training_gpus = 1

    def setup(self) -> None:
        tokenizer_kwargs = {
            "trust_remote_code": True,
            "use_fast": not self.cfg.trainer.disable_fast_tokenizer,
            "padding_side": "left",
        }
        self.is_vlm = False
        self.tokenizer = get_tokenizer(self.cfg.trainer.policy.model.path, **tokenizer_kwargs)
        self.collator = self._build_collator(self.tokenizer)
        self._init_tracker()
        self._init_workers()

    def _init_tracker(self) -> None:
        self.tracker = None

    def _init_workers(self) -> None:
        if self.is_vlm:
            raise ValueError("Fireworks SFT currently supports text-only models")
        runtime = FireworksRuntime.connect(
            config=self.sft_cfg.fireworks,
            tokenizer=self.tokenizer,
            tokenizer_model=self.sft_cfg.model.path,
            lora_rank=self.sft_cfg.model.lora.rank,
            learning_rate=self.sft_cfg.optimizer_config.lr,
            create_deployment=False,
        )
        self._fireworks_runtime = runtime
        try:
            self.dispatch = FireworksSFTDispatch(
                runtime,
                self.sft_cfg.fireworks,
                self.sft_cfg.optimizer_config,
            )
            super()._init_tracker()
        except BaseException:
            asyncio.run(runtime.close())
            self._fireworks_runtime = None
            raise

    def _validate_batch_parallelism(self) -> None:
        if self.sft_cfg.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def export_final_model(self) -> dict | None:
        output_model_id = self.sft_cfg.fireworks.output_model_id
        if not output_model_id:
            return None
        if self._fireworks_runtime is None:
            raise RuntimeError("Fireworks runtime is not initialized")

        step_dir = os.path.join(self.sft_cfg.ckpt_path, f"global_step_{self.global_step}")
        dcp_manifest_path = os.path.join(step_dir, "policy", "fireworks_checkpoint.json")
        with io.open_file(dcp_manifest_path, "r") as f:
            dcp_manifest = json.load(f)
        checkpoint_name = str(dcp_manifest.get("checkpoint_name") or "")
        if not checkpoint_name:
            raise ValueError(f"Fireworks DCP manifest has no checkpoint_name: {dcp_manifest_path}")

        promotable = dcp_manifest.get("promotable_checkpoint")
        snapshot_path = promotable.get("snapshot_path") if isinstance(promotable, dict) else None
        checkpoint_resource = promotable.get("checkpoint_resource") if isinstance(promotable, dict) else None
        if (
            isinstance(snapshot_path, str)
            and snapshot_path
            and isinstance(checkpoint_resource, str)
            and "/checkpoints/" in checkpoint_resource
        ):
            result = asyncio.run(
                self._fireworks_runtime.promote_checkpoint_resource(
                    checkpoint=PromotableCheckpoint(
                        snapshot_path=snapshot_path,
                        checkpoint_resource=checkpoint_resource,
                        checkpoint_type=promotable.get("checkpoint_type"),
                    ),
                    output_model_id=output_model_id,
                )
            )
        else:
            result = asyncio.run(
                self._fireworks_runtime.promote_final_model(
                    checkpoint_name=checkpoint_name,
                    output_model_id=output_model_id,
                )
            )
        manifest = {
            "format_version": 1,
            "global_step": self.global_step,
            "trainer_job_id": self._fireworks_runtime.trainer_job_id,
            "dcp_checkpoint": dcp_manifest,
            **result,
        }
        export_manifest_path = os.path.join(step_dir, "fireworks_final_model.json")
        with io.open_file(export_manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")

        if self.sft_cfg.fireworks.delete_trainer_after_promotion:
            trainer_state = asyncio.run(self._fireworks_runtime.delete_trainer())
            manifest["trainer_state_after_delete"] = trainer_state
            cleanup_manifest_path = os.path.join(step_dir, "fireworks_trainer_cleanup.json")
            with io.open_file(cleanup_manifest_path, "w") as f:
                json.dump(
                    {
                        "trainer_job_id": self._fireworks_runtime.trainer_job_id,
                        "state": trainer_state,
                    },
                    f,
                    indent=2,
                    sort_keys=True,
                )
                f.write("\n")
        return manifest

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            if self._fireworks_runtime is not None:
                asyncio.run(self._fireworks_runtime.close())
                self._fireworks_runtime = None
