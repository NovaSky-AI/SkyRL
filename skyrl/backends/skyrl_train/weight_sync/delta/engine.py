"""vLLM receive-side engine for SkyRL checkpoint-delta weight sync.

Imports vLLM at module scope, so import it lazily from anything that must work
without the wheel. Nothing does: ``weight_sync/register.py`` names this module
by path and vLLM imports it only when a worker builds the ``delta`` backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch
from vllm.distributed.weight_transfer.base import WeightTransferEngine
from vllm.logger import init_logger

from skyrl.backends.skyrl_train.weight_sync.delta.checkpoint import (
    LocalCheckpointStore,
)
from skyrl.backends.skyrl_train.weight_sync.weight_receivers import (
    empty_cuda_cache_rocm,
)

logger = init_logger(__name__)


@dataclass
class DeltaTransferInitInfo:
    base_model_path: str
    local_checkpoint_dir: str
    cloud_download_workers: int = 4
    checkpoint_load_format: str = "vllm_multi_thread_safetensors"
    multi_thread_safetensors_max_workers: int = 8


@dataclass
class DeltaTransferUpdateInfo:
    target_version: int
    sync_dir: str | None = None
    uri: str | None = None
    version: int | None = None

    @property
    def resolved_target_version(self) -> int:
        version = self.target_version if self.target_version is not None else self.version
        if version is None:
            raise ValueError("Delta update_info requires target_version")
        return int(version)


class DeltaWeightTransferEngine(WeightTransferEngine[DeltaTransferInitInfo, DeltaTransferUpdateInfo]):
    """Receive compressed checkpoint deltas and load updated weights into vLLM.

    The base supplies ``__init__``, the draft-session retarget hooks
    (``set_weight_update_target`` / ``reset_weight_update_target``) and the
    ``_default_model`` bookkeeping they restore from; only the delta-specific
    lifecycle is implemented here.
    """

    init_info_cls = DeltaTransferInitInfo
    update_info_cls = DeltaTransferUpdateInfo

    # Read by vLLM's Worker._start_weight_update when updating a draft model.
    # Delta sync reloads the full checkpoint for the target model only.
    supports_draft_weight_update = False

    def __init__(self, config: Any, vllm_config: Any, device: Any, model: torch.nn.Module) -> None:
        super().__init__(config, vllm_config, device, model)
        self._store: LocalCheckpointStore | None = None
        self._checkpoint_load_format = "vllm_multi_thread_safetensors"
        self._multi_thread_safetensors_max_workers = 8
        self._cloud_download_workers = 4

    def start_weight_update(self) -> None:
        """Initialize layerwise reloading for the incoming checkpoint weights."""
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

        with torch.device(self.device):
            initialize_layerwise_reload(self.model)

    def finish_weight_update(self) -> None:
        """Finalize layerwise reloading after the checkpoint has been loaded."""
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        with torch.device(self.device):
            finalize_layerwise_reload(self.model, self.model_config)
        empty_cuda_cache_rocm()

    def update_weights(self, update_info: dict[str, Any]) -> None:
        """Load one update, as vLLM's native ``/update_weights`` endpoint expects."""
        self.receive_weights(self.parse_update_info(update_info))
        torch.accelerator.synchronize()

    def parse_init_info(self, init_dict: dict[str, Any]) -> DeltaTransferInitInfo:
        try:
            return self.init_info_cls(**init_dict)
        except TypeError as e:
            raise ValueError(f"Invalid init_info for {self.__class__.__name__}: {e}") from e

    def parse_update_info(self, update_dict: dict[str, Any]) -> DeltaTransferUpdateInfo:
        try:
            allowed = set(self.update_info_cls.__dataclass_fields__.keys())
            return self.update_info_cls(**{k: v for k, v in update_dict.items() if k in allowed})
        except TypeError as e:
            raise ValueError(f"Invalid update_info for {self.__class__.__name__}: {e}") from e

    def init_transfer_engine(self, init_info: DeltaTransferInitInfo) -> None:
        self._store = LocalCheckpointStore(
            base_model_path=init_info.base_model_path,
            local_checkpoint_dir=init_info.local_checkpoint_dir,
            cloud_download_workers=init_info.cloud_download_workers,
        )
        self._checkpoint_load_format = init_info.checkpoint_load_format
        self._multi_thread_safetensors_max_workers = init_info.multi_thread_safetensors_max_workers
        self._cloud_download_workers = init_info.cloud_download_workers
        logger.info(
            "Initialized delta weight transfer engine: base_model_path=%s local_checkpoint_dir=%s "
            "checkpoint_load_format=%s cloud_download_workers=%s",
            init_info.base_model_path,
            init_info.local_checkpoint_dir,
            self._checkpoint_load_format,
            self._cloud_download_workers,
        )

    def fetch_weights(self, target_version: int, sync_dir: str | None = None, uri: str | None = None) -> dict[str, Any]:
        if self._store is None:
            raise RuntimeError("DeltaWeightTransferEngine has not been initialized")
        t0 = time.perf_counter()
        stats = self._store.fetch(target_version=target_version, sync_dir=sync_dir, uri=uri)
        total_s = time.perf_counter() - t0
        fetch_s, apply_s, reset_s = stats.get("fetch_s", 0.0), stats.get("apply_s", 0.0), stats.get("reset_s", 0.0)
        message = (
            f"delta checkpoint fetch: target_version={target_version} fetch_s={fetch_s:.3f} "
            f"apply_s={apply_s:.3f} reset_s={reset_s:.3f} total_s={total_s:.3f}"
        )
        logger.info(message)
        print(message, flush=True)
        return {"status": "ok", "target_version": target_version, "stats": {**stats, "total_s": total_s}}

    def receive_weights(self, update_info: DeltaTransferUpdateInfo) -> None:
        if self._store is None:
            raise RuntimeError("DeltaWeightTransferEngine has not been initialized")

        t0 = time.perf_counter()
        target_version = update_info.resolved_target_version
        self._store.validate_ready(target_version)
        prepare_s = time.perf_counter() - t0
        load_s = 0.0
        t1 = time.perf_counter()

        def tensors():
            return self._store.iter_tensors(
                load_format=self._checkpoint_load_format,
                multi_thread_safetensors_max_workers=self._multi_thread_safetensors_max_workers,
            )

        # MTP architectures raise on incomplete layer coverage.
        from vllm.model_executor.model_loader.mtp_validation import (
            disable_mtp_completeness_check,
        )

        with torch.device(self.device), disable_mtp_completeness_check():
            self.model.load_weights(tensors())
            # The spec-decode drafter is a separate module the main load never
            # touches. Unlike the push backends this does NOT go through
            # skyrl_drafter_reload's proxy: that materializes the weight list so
            # the drafter can re-read it, which for a whole checkpoint would be
            # the entire model resident at once. The store re-streams instead.
            self._reload_drafter(tensors)

        load_s = time.perf_counter() - t1
        total_s = time.perf_counter() - t0
        message = (
            "delta checkpoint receive reload-only: target_version=%s checkpoint_load_format=%s "
            "prepare_s=%.3f load_s=%.3f total_s=%.3f"
        ) % (
            target_version,
            self._checkpoint_load_format,
            prepare_s,
            load_s,
            total_s,
        )
        logger.info(message)
        print(message, flush=True)

    def _reload_drafter(self, tensors) -> None:
        """Reload the spec-decode drafter from a fresh pass over the checkpoint.

        No-op, and no second pass, when this process has no loadable proposer.
        """
        from skyrl.backends.skyrl_train.patches.vllm.patch_model_runner_registry import (
            current_model_runner,
        )

        model_runner = current_model_runner()
        drafter = getattr(model_runner, "drafter", None) if model_runner is not None else None
        if drafter is None or getattr(drafter, "model", None) is None:
            return

        from skyrl.backends.skyrl_train.inference_servers.spec_decode_utils import (
            _reload_spec_decode_drafter,
        )

        # NOTE: This only works for MTP because we iterate over the target model's tensors
        _reload_spec_decode_drafter(model_runner, tensors())

    def shutdown(self):
        self._store = None
