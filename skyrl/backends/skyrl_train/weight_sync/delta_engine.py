"""vLLM receive-side engine for SkyRL checkpoint-delta weight sync."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Iterator

import torch

from skyrl.backends.skyrl_train.weight_sync.delta_checkpoint import (
    LocalCheckpointStore,
)

try:
    from vllm.logger import init_logger

    logger = init_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)


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
    # vLLM 0.23's native /update_weights path checks this attribute before it
    # dispatches into the custom transfer engine. Delta checkpoint sync always
    # reloads dense prepared checkpoint tensors.
    update_kind: str = "dense"

    @property
    def resolved_target_version(self) -> int:
        version = self.target_version if self.target_version is not None else self.version
        if version is None:
            raise ValueError("Delta update_info requires target_version")
        return int(version)


def register_delta_weight_transfer_engine() -> None:
    """Register SkyRL's custom vLLM weight-transfer backend under ``delta``."""
    from vllm.distributed.weight_transfer.factory import WeightTransferEngineFactory

    try:
        WeightTransferEngineFactory.register_engine(
            "delta",
            "skyrl.backends.skyrl_train.weight_sync.delta_engine",
            "DeltaWeightTransferEngine",
        )
    except ValueError as e:
        if "already registered" not in str(e):
            raise


class DeltaWeightTransferEngine:
    """Receive compressed checkpoint deltas and load updated weights into vLLM."""

    init_info_cls = DeltaTransferInitInfo
    update_info_cls = DeltaTransferUpdateInfo

    # Read by vLLM's Worker._start_weight_update when updating a draft model.
    # Delta sync reloads the full checkpoint for the target model only.
    supports_draft_weight_update = False

    def __init__(self, config: Any, vllm_config: Any, device: Any, model: torch.nn.Module) -> None:
        # Signature mirrors vLLM's WeightTransferEngine base (0.26+):
        # WeightTransferEngineFactory.create_engine calls
        # engine_cls(config, vllm_config, device, model). Duck-typed rather than
        # subclassed so this module stays importable without vLLM installed.
        self.config = config
        self.vllm_config = vllm_config
        self.parallel_config = getattr(vllm_config, "parallel_config", None)
        self.model_config = getattr(vllm_config, "model_config", None)
        self.device = device
        self.model = model
        self._default_model = model
        self._default_model_config = self.model_config
        self._store: LocalCheckpointStore | None = None
        self._checkpoint_load_format = "vllm_multi_thread_safetensors"
        self._multi_thread_safetensors_max_workers = 8
        self._cloud_download_workers = 4

    def set_weight_update_target(self, model: Any, model_config: Any) -> None:
        """Retarget the active weight update (see WeightTransferEngine base)."""
        self.model = model
        self.model_config = model_config

    def reset_weight_update_target(self) -> None:
        """Restore weight updates to the default target model."""
        self.model = self._default_model
        self.model_config = self._default_model_config

    def start_weight_update(self) -> None:
        """No-op: SkyRL drives the layerwise-reload lifecycle from the worker.

        vLLM's ``Worker.start_weight_update`` delegates here, but SkyRL's delta
        flow calls ``NewInferenceWorkerWrap.skyrl_start_weight_update`` instead
        (see DeltaWeightTransferSender._apply_receiver_update), which is what
        initializes layerwise reload. Doing it again here would double-initialize.
        """

    def finish_weight_update(self) -> None:
        """No-op counterpart to :meth:`start_weight_update`.

        SkyRL finalizes layerwise reload via
        ``NewInferenceWorkerWrap.skyrl_finish_weight_update``.
        """

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
        message = f"delta checkpoint fetch: target_version={target_version} fetch_s={fetch_s:.3f} apply_s={apply_s:.3f}  reset_s={reset_s:.3f} total_s={total_s:.3f}"
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
        self.model.load_weights(
            self._store.iter_tensors(
                load_format=self._checkpoint_load_format,
                multi_thread_safetensors_max_workers=self._multi_thread_safetensors_max_workers,
            )
        )
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

    def shutdown(self):
        self._store = None

    @staticmethod
    def trainer_send_weights(
        _iterator: Iterator[tuple[str, torch.Tensor]], _trainer_args: dict[str, Any] | Any
    ) -> None:
        raise NotImplementedError("Delta weight sync publishes through SkyRL's DeltaWeightTransferSender")
