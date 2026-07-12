"""vLLM receive-side engine for SkyRL checkpoint-delta weight sync."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator

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
    max_files_to_keep: int = 5
    prefetch_depth: int = 0
    version_wait_timeout_s: float = 7200.0


@dataclass
class DeltaTransferUpdateInfo:
    target_version: int | None = None
    sync_dir: str | None = None
    uri: str | None = None
    version: int | None = None
    update_kind: str = "dense"
    receive_update_kind: str | None = None

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

    def __init__(self, config: Any, parallel_config: Any, model: torch.nn.Module) -> None:
        self.config = config
        self.parallel_config = parallel_config
        self.model = model
        self._store: LocalCheckpointStore | None = None

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
        )
        logger.info(
            "Initialized delta weight transfer engine: base_model_path=%s local_checkpoint_dir=%s",
            init_info.base_model_path,
            init_info.local_checkpoint_dir,
        )

    def fetch_weights(self, target_version: int, sync_dir: str | None = None, uri: str | None = None) -> dict[str, Any]:
        if self._store is None:
            raise RuntimeError("DeltaWeightTransferEngine has not been initialized")
        t0 = time.perf_counter()
        stats = self._store.fetch(target_version=target_version, sync_dir=sync_dir, uri=uri)
        total_s = time.perf_counter() - t0
        message = ("delta checkpoint fetch: target_version=%s fetch_s=%.3f apply_s=%.3f reset_s=%.3f total_s=%.3f") % (
            target_version,
            stats.get("fetch_s", 0.0),
            stats.get("apply_s", 0.0),
            stats.get("reset_s", 0.0),
            total_s,
        )
        logger.info(message)
        print(message, flush=True)
        return {"status": "ok", "target_version": target_version, "stats": {**stats, "total_s": total_s}}

    def receive_weights(
        self,
        update_info: DeltaTransferUpdateInfo,
        load_weights: Callable[[Iterator[tuple[str, torch.Tensor]]], None],
    ) -> None:
        if self._store is None:
            raise RuntimeError("DeltaWeightTransferEngine has not been initialized")

        t0 = time.perf_counter()
        target_version = update_info.resolved_target_version
        self._store.validate_ready(target_version)
        prepare_s = time.perf_counter() - t0
        load_s = 0.0
        t1 = time.perf_counter()
        load_weights(self._store.iter_tensors())
        load_s = time.perf_counter() - t1
        total_s = time.perf_counter() - t0
        message = (
            "delta checkpoint receive reload-only: target_version=%s prepare_s=%.3f load_s=%.3f total_s=%.3f"
        ) % (
            target_version,
            prepare_s,
            load_s,
            total_s,
        )
        logger.info(message)
        print(message, flush=True)

    def shutdown(self) -> None:
        self._store = None

    @staticmethod
    def trainer_send_weights(
        _iterator: Iterator[tuple[str, torch.Tensor]], _trainer_args: dict[str, Any] | Any
    ) -> None:
        raise NotImplementedError("Delta weight sync publishes through SkyRL's DeltaWeightTransferSender")
