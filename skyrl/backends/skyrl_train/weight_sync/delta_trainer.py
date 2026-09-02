"""Trainer-side engine for the checkpoint-delta weight-sync backend.

The trainer publishes a compressed delta against the base checkpoint to disk or
object storage; each inference worker fetches and reloads it. Nothing crosses the
network fabric, but the shape is vLLM's trainer-send: a ``WeightSource`` in, a
four-method ``VLLMWeightSyncClient`` out, ``send_weights()`` owning the round
trip.

Two parts of that round trip ride the engine rather than the client protocol:

* ``fetch_weights`` runs **before** the pause so the download overlaps live
  generation.
* the pause / resume bracket and the prefix-cache reset. This is the only backend
  that reloads a whole checkpoint in place, so the only one that needs generation
  stopped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Optional

import torch
from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
)

from skyrl.backends.skyrl_train.weight_sync.delta_checkpoint import (
    DeltaCheckpointPublisher,
    DeltaPublishResult,
)

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)

DELTA_BACKEND = "delta"


@dataclass
class DeltaTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for checkpoint-delta weight sync.

    ``base_model_path``, ``local_checkpoint_dir``, the load format and its worker
    counts are propagated to the worker at ``trainer_init``; the rest are
    publisher-side only.
    """

    backend: ClassVar[str] = DELTA_BACKEND

    base_model_path: str
    sync_dir: str
    local_checkpoint_dir: str
    publish_staging_dir: str
    max_file_size_in_gb: float = 1.0
    cloud_download_workers: int = 4
    publish_num_workers: Optional[int] = None
    checkpoint_load_format: str = "vllm_multi_thread_safetensors"
    multi_thread_safetensors_max_workers: int = 8


class DeltaTrainerWeightTransferEngine(TrainerWeightTransferEngine[DeltaTrainerInitInfo]):
    """Publish a checkpoint delta and drive the inference-side reload."""

    init_info_cls = DeltaTrainerInitInfo

    # Resets the prefix cache itself, inside the pause bracket, so the worker
    # must not also fire a concurrent reset (see workers/worker.py).
    skyrl_handles_prefix_cache_reset = True

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        init_info: DeltaTrainerInitInfo,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self._init_info = init_info
        self._publisher: Optional[DeltaCheckpointPublisher] = None
        self._reset_prefix_cache = False

    @classmethod
    def trainer_init(
        cls,
        init_info: DeltaTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: Optional[WeightSource] = None,
    ) -> "Self":
        if source is None:
            raise ValueError("Delta trainer weight transfer requires a WeightSource.")
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            init_info=init_info,
        )
        if engine.is_sender:
            # No data-plane rendezvous; this builds each worker's local
            # checkpoint store and records the load format.
            engine.client.init_weight_transfer_engine(
                {
                    "base_model_path": init_info.base_model_path,
                    "local_checkpoint_dir": init_info.local_checkpoint_dir,
                    "cloud_download_workers": init_info.cloud_download_workers,
                    "checkpoint_load_format": init_info.checkpoint_load_format,
                    "multi_thread_safetensors_max_workers": init_info.multi_thread_safetensors_max_workers,
                }
            )
        return engine

    def skyrl_set_reset_prefix_cache(self, reset: bool) -> None:
        """Whether this round should reset the prefix cache inside the pause."""
        self._reset_prefix_cache = bool(reset)

    def send_weights(self) -> None:
        """Publish this round's delta and reload it on every inference worker.

        Called on **every** trainer rank: draining the source drives its gather
        collectives, so a rank that skipped it would hang its peers. Only rank 0
        stages and uploads payloads, and only rank 0 drives the inference side.
        """
        assert self.source is not None  # guaranteed by trainer_init
        distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if distributed else 0

        if self._publisher is None:
            self._publisher = DeltaCheckpointPublisher(
                base_model_path=self._init_info.base_model_path,
                sync_dir=self._init_info.sync_dir,
                publish_staging_dir=self._init_info.publish_staging_dir,
                max_file_size_in_gb=self._init_info.max_file_size_in_gb,
                publish_num_workers=self._init_info.publish_num_workers,
            )

        local_result = self._publisher.create_delta_files(self.source)
        if not isinstance(local_result, DeltaPublishResult):
            raise TypeError(f"Expected DeltaPublishResult from the publisher, got {type(local_result)}")

        if distributed:
            world_size = torch.distributed.get_world_size()
            gathered: list[Optional[DeltaPublishResult]] = [None] * world_size
            torch.distributed.all_gather_object(gathered, local_result)
        else:
            gathered = [local_result]

        update_info = None
        if rank == 0:
            source_results = [r for r in gathered if r is not None and r.rank == 0]
            update_info = self._publisher.publish(source_results)

        if distributed:
            box = [update_info]
            torch.distributed.broadcast_object_list(box, src=0)
            update_info = box[0]

        if self.is_sender and update_info is not None:
            self._apply_receiver_update(update_info)

    def _apply_receiver_update(self, update_info: dict) -> None:
        target_version = int(update_info.get("target_version", update_info.get("version")))
        # Before the pause, so the download overlaps live generation.
        self.client.fetch_weights(
            target_version=target_version,
            sync_dir=update_info.get("sync_dir", self._init_info.sync_dir),
            uri=update_info.get("uri"),
        )
        self.client.pause_generation()
        try:
            if self._reset_prefix_cache:
                self.client.reset_prefix_cache(reset_running_requests=True)
            self.client.start_weight_update()
            self.client.update_weights(update_info)
            self.client.finish_weight_update()
        finally:
            self.client.resume_generation()

    def shutdown(self) -> None:
        self._publisher = None


def register_delta_trainer_engine() -> None:
    """Register ``delta`` in ``WeightTransferTrainerFactory`` (idempotent)."""
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory

    if DELTA_BACKEND in WeightTransferTrainerFactory._registry:
        return
    WeightTransferTrainerFactory.register_engine(
        DELTA_BACKEND,
        "skyrl.backends.skyrl_train.weight_sync.delta_trainer",
        "DeltaTrainerWeightTransferEngine",
    )
