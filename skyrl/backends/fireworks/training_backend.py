"""Trainer-facing dispatch adapter for Fireworks hosted GRPO."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Optional

from loguru import logger

from skyrl.backends.fireworks.grpo import build_tinker_grpo_datums
from skyrl.backends.fireworks.runtime import FireworksRuntime
from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.train.config import SkyRLTrainConfig


class FireworksPolicyDispatch:
    """Policy-only subset of ``WorkerDispatch`` backed by Fireworks APIs."""

    def __init__(
        self,
        cfg: SkyRLTrainConfig,
        runtime: FireworksRuntime,
        *,
        datum_builder=build_tinker_grpo_datums,
    ) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self._datum_builder = datum_builder
        self._last_optim_metrics: dict[str, float] = {}

    def get_lcm_dp_size(self) -> int:
        return 1

    def dp_size(self, model: str) -> int:
        self._require_policy(model)
        return 1

    @staticmethod
    def _require_policy(model: str) -> None:
        if model != "policy":
            raise NotImplementedError(
                f"Fireworks GRPO is policy-only, got model={model!r}"
            )

    def stage_data(self, model: str, data: TrainingInputBatch, mini_batch_boundaries):
        self._require_policy(model)
        return [data[start:end] for start, end in mini_batch_boundaries]

    def forward_backward_from_staged(
        self,
        model: str,
        staged_batch: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[dict[str, Any]] = None,
        model_id: Optional[str] = None,
    ) -> WorkerOutput:
        self._require_policy(model)
        if loss_fn is not None or loss_fn_config is not None or model_id is not None:
            raise NotImplementedError(
                "Fireworks native GRPO dispatch does not accept per-call loss/model overrides"
            )
        datums = self._datum_builder(
            staged_batch,
            max_seq_len=self.cfg.trainer.fireworks.max_seq_len,
        )
        result = self.runtime.training_client.forward_backward(
            datums, "importance_sampling"
        ).result(timeout=self.cfg.trainer.fireworks.request_timeout_s)
        metrics = {
            key: float(value)
            for key, value in (getattr(result, "metrics", None) or {}).items()
        }
        if "loss:sum" in metrics:
            metrics.setdefault("final_loss", metrics["loss:sum"])
        return WorkerOutput(
            loss_fn_output_type=str(getattr(result, "loss_fn_output_type", "scalar")),
            loss_fn_outputs=[],
            metrics=metrics,
        )

    def optim_step(self, model: str, model_id: Optional[str] = None) -> Optional[float]:
        self._require_policy(model)
        if model_id is not None:
            raise NotImplementedError(
                "Fireworks GRPO does not support model_id overrides"
            )
        try:
            import tinker
        except ImportError as exc:
            raise ImportError(
                "Fireworks optimizer construction requires the tinker package"
            ) from exc

        optimizer = self.cfg.trainer.policy.optimizer_config
        params = tinker.AdamParams(
            learning_rate=optimizer.lr,
            beta1=optimizer.adam_betas[0],
            beta2=optimizer.adam_betas[1],
            eps=self.cfg.trainer.fireworks.adam_eps,
            weight_decay=optimizer.weight_decay,
            grad_clip_norm=optimizer.max_grad_norm,
        )
        result = self.runtime.training_client.optim_step(params).result(
            timeout=self.cfg.trainer.fireworks.request_timeout_s
        )
        self._last_optim_metrics = {
            key: float(value)
            for key, value in (getattr(result, "metrics", None) or {}).items()
        }
        for key, value in self._last_optim_metrics.items():
            if "grad_norm" in key:
                return value
        return None

    async def save_weights_for_sampler(self, model_id: Optional[str] = None) -> None:
        if model_id is not None:
            raise NotImplementedError(
                "Fireworks GRPO does not support model_id overrides"
            )
        identity = await self.runtime.publish_sampler_weights()
        logger.info(
            "Published Fireworks sampler weights: version={}, snapshot={}",
            identity.version,
            identity.snapshot_path,
        )

    def init_weight_sync_state(self, inference_engine_client) -> None:
        if getattr(inference_engine_client, "runtime", None) is not self.runtime:
            raise ValueError(
                "Fireworks training and inference adapters must share one runtime"
            )

    def empty_cache(self, model: Optional[str] = None) -> None:
        if model is not None:
            self._require_policy(model)

    def mark_all_offloaded(self) -> None:
        return None

    def get_node_ids(self) -> list[str]:
        return []

    def start_profile(self, model: str) -> None:
        self._require_policy(model)

    def profile_step(self, model: str) -> None:
        self._require_policy(model)

    def stop_profile(self, model: str) -> None:
        self._require_policy(model)

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Fireworks GRPO requires rollout logprobs so the local policy forward is skipped"
        )

    def forward_from_staged(self, *args, **kwargs):
        raise NotImplementedError(
            "Fireworks GRPO requires rollout logprobs so the local policy forward is skipped"
        )

    _CHECKPOINT_MANIFEST = "fireworks_checkpoint.json"

    def save_checkpoint(self, model: str, ckpt_dir: str, tokenizer=None) -> None:
        """Save persistent Fireworks DCP state and a small local resume manifest.

        ``save_state`` is deliberately separate from the sampler snapshots used
        for per-step hot-loads: DCP includes both weights and optimizer state.
        SkyRL owns the local trainer/dataloader checkpoint alongside this
        manifest, while Fireworks owns the large model checkpoint remotely.
        """

        self._require_policy(model)
        del tokenizer  # Tokenizer files are unchanged and already identified by the config.

        step_match = re.search(r"global_step_(\d+)", ckpt_dir)
        step = step_match.group(1) if step_match else "unknown"
        checkpoint_name = f"skyrl-step-{step}-{uuid.uuid4().hex[:8]}"
        result = self.runtime.training_client.save_state(checkpoint_name).result(
            timeout=self.cfg.trainer.fireworks.request_timeout_s
        )
        provider_path = str(getattr(result, "path", "") or "")
        if not provider_path:
            raise RuntimeError(
                f"Fireworks save_state({checkpoint_name!r}) returned no checkpoint path"
            )

        manifest = {
            "format_version": 1,
            "checkpoint_kind": "fireworks_dcp",
            "checkpoint_name": checkpoint_name,
            "provider_path": provider_path,
            "source_trainer_job_id": self.runtime.trainer_job_id,
            "includes_optimizer_state": True,
        }
        io.makedirs(ckpt_dir, exist_ok=True)
        manifest_path = os.path.join(ckpt_dir, self._CHECKPOINT_MANIFEST)
        with io.open_file(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        logger.info(
            "Saved Fireworks DCP checkpoint: name={}, path={}, manifest={}",
            checkpoint_name,
            provider_path,
            manifest_path,
        )

    def load_checkpoint(
        self,
        model: str,
        ckpt_dir: str,
        *,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ) -> None:
        """Restore a Fireworks DCP checkpoint, including optimizer by default."""

        self._require_policy(model)
        if load_lr_scheduler_states and not load_optimizer_states:
            raise ValueError(
                "Fireworks cannot restore scheduler state without the optimizer state"
            )

        manifest_path = os.path.join(ckpt_dir, self._CHECKPOINT_MANIFEST)
        if not io.exists(manifest_path):
            raise FileNotFoundError(
                f"Fireworks checkpoint manifest not found: {manifest_path}"
            )
        with io.open_file(manifest_path, "r") as f:
            manifest = json.load(f)
        if manifest.get("format_version") != 1:
            raise ValueError(
                "Unsupported Fireworks checkpoint manifest version: "
                f"{manifest.get('format_version')!r}"
            )

        checkpoint_name = str(manifest.get("checkpoint_name") or "")
        provider_path = str(manifest.get("provider_path") or "")
        source_job_id = manifest.get("source_trainer_job_id")
        if source_job_id and checkpoint_name:
            load_path = self.runtime.training_client.resolve_checkpoint_path(
                checkpoint_name,
                source_job_id=str(source_job_id),
            )
        elif provider_path:
            # Serverless sessions may not expose a dedicated trainer job ID;
            # their returned Tinker path is already directly loadable.
            load_path = provider_path
        else:
            raise ValueError(
                f"Fireworks checkpoint manifest has no loadable reference: {manifest_path}"
            )

        load = (
            self.runtime.training_client.load_state_with_optimizer
            if load_optimizer_states
            else self.runtime.training_client.load_state
        )
        load(load_path).result(timeout=self.cfg.trainer.fireworks.request_timeout_s)
        logger.info(
            "Loaded Fireworks DCP checkpoint: reference={}, optimizer_restored={}",
            load_path,
            load_optimizer_states,
        )

    def save_hf_model(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "Promoting a Fireworks adapter is not implemented yet"
        )
