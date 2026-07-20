"""Disk/cloud checkpoint-delta transfer strategy."""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

import torch

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk
from skyrl.backends.skyrl_train.weight_sync.delta_checkpoint import (
    SUPPORTED_CHECKPOINT_LOAD_FORMATS,
    DeltaCheckpointPublisher,
    DeltaPublishResult,
)
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferSender,
    WeightTransferStrategy,
)
from skyrl.backends.skyrl_train.weight_sync.weight_extractor import ExtractorShardInfo

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.train.config import InferenceEngineConfig


logger = logging.getLogger(__name__)


@dataclass
class DeltaInitInfo(WeightSyncInitInfo):
    base_model_path: str
    sync_dir: str
    local_checkpoint_dir: str
    publish_staging_dir: Optional[str] = None
    max_file_size_in_gb: float = 1.0
    gcs_download_workers: int = 4
    publish_num_workers: Optional[int] = None
    checkpoint_load_format: str = "vllm_fastsafetensors"
    multi_thread_safetensors_max_workers: int = 8

    def for_servers(self, world_size_per_server: int, num_servers: int, dp_size: int = 1) -> List["DeltaInitInfo"]:
        return [copy.deepcopy(self) for _ in range(num_servers)]

    def to_api_payload(self) -> Dict[str, Any]:
        return {
            "base_model_path": self.base_model_path,
            "local_checkpoint_dir": self.local_checkpoint_dir,
            "gcs_download_workers": self.gcs_download_workers,
            "checkpoint_load_format": self.checkpoint_load_format,
            "multi_thread_safetensors_max_workers": self.multi_thread_safetensors_max_workers,
        }


class DeltaWeightTransferSender(WeightTransferSender):
    handles_prefix_cache_reset = True

    def __init__(self, init_info: DeltaInitInfo, inference_client: "RemoteInferenceClient") -> None:
        self._init_info = init_info
        self._inference_client = inference_client
        self._publisher: Optional[DeltaCheckpointPublisher] = None
        self._seed_complete = False

    async def _apply_receiver_update(self, update_info: Dict[str, Any], rank: int) -> None:
        if update_info.get("noop", False):
            return

        target_version = int(update_info.get("target_version", update_info.get("version")))
        await self._inference_client.fetch_weights(
            target_version=target_version,
            sync_dir=update_info.get("sync_dir", self._init_info.sync_dir),
            uri=update_info.get("uri"),
        )
        await self._inference_client.pause_generation()
        try:
            await self._inference_client.reset_prefix_cache(reset_running_requests=True)
            await self._inference_client.start_weight_update(is_checkpoint_format=True)
            await self._inference_client.update_named_weights(update_info)
            await self._inference_client.finish_weight_update()
        finally:
            await self._inference_client.resume_generation()

    async def send_chunks(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
        extractor_shard_info: Optional[ExtractorShardInfo] = None,
    ) -> None:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        if self._publisher is None:
            self._publisher = DeltaCheckpointPublisher(
                base_model_path=self._init_info.base_model_path,
                sync_dir=self._init_info.sync_dir,
                publish_staging_dir=self._init_info.publish_staging_dir,
                max_file_size_in_gb=self._init_info.max_file_size_in_gb,
                publish_num_workers=self._init_info.publish_num_workers,
            )
        if extractor_shard_info is None:
            extractor_shard_info = self._publisher._default_shard_info()

        if not self._seed_complete:
            self._seed_complete = True
            if rank == 0:
                logger.info(
                    "delta checkpoint seed sync: treating base_model_path as version 0; "
                    "skipping chunk extraction and delta publish"
                )
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            return

        source_shard_info = replace(
            extractor_shard_info,
            is_source_rank=(rank == 0),
            replicate_world_size=1,
            source_index_in_replicate_world=0,
            rank=rank,
        )

        local_result = await asyncio.to_thread(self._publisher.publish, chunks, source_shard_info)
        if not isinstance(local_result, DeltaPublishResult):
            raise TypeError(f"Expected DeltaPublishResult from sharded publisher, got {type(local_result)}")

        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        world_size = torch.distributed.get_world_size() if is_distributed else 1

        if is_distributed:
            gathered_results: list[Optional[DeltaPublishResult]] = [None] * world_size
            torch.distributed.all_gather_object(gathered_results, local_result)
        else:
            gathered_results = [local_result]

        update_info = None
        if rank == 0:
            source_results = [result for result in gathered_results if result is not None and result.rank == 0]
            update_info = self._publisher.finalize_publish(source_results)

        if is_distributed:
            update_info_box = [update_info]
            torch.distributed.broadcast_object_list(update_info_box, src=0)
            update_info = update_info_box[0]

        control_rank = 1 if world_size > 1 else 0
        if rank == control_rank and update_info is not None and not update_info.get("noop", False):
            await self._apply_receiver_update(update_info, rank)
        # The worker runs a process-group barrier after send_chunks returns. Every rank
        # reaches it (non-control ranks return immediately; the control rank returns once
        # the receiver-side reload finishes), so the barrier simply syncs all ranks on the
        # slowest one -- which is what WorkerDispatch already waits for.

    def teardown(self) -> None:
        self._publisher = None


class DeltaTransferStrategy(WeightTransferStrategy):
    @staticmethod
    def create_init_info(
        ie_cfg: "InferenceEngineConfig",
        inference_world_size: Optional[int] = None,
        base_model_path: Optional[str] = None,
    ) -> DeltaInitInfo:
        if base_model_path is None:
            raise ValueError("Delta weight sync requires base_model_path")
        delta_cfg = ie_cfg.delta_weight_sync
        if not delta_cfg.sync_dir:
            raise ValueError("Delta weight sync requires generator.inference_engine.delta_weight_sync.sync_dir")

        from skyrl.backends.skyrl_train.weight_sync.delta_checkpoint import (
            _safe_path_name,
        )

        safe_model_name = _safe_path_name(base_model_path)
        local_checkpoint_dir = delta_cfg.local_checkpoint_dir or f"/tmp/skyrl_delta_checkpoints/{safe_model_name}"
        if delta_cfg.checkpoint_load_format not in SUPPORTED_CHECKPOINT_LOAD_FORMATS:
            raise ValueError(
                "Delta checkpoint_load_format must be one of "
                f"{sorted(SUPPORTED_CHECKPOINT_LOAD_FORMATS)}, got {delta_cfg.checkpoint_load_format!r}"
            )
        return DeltaInitInfo(
            base_model_path=base_model_path,
            sync_dir=delta_cfg.sync_dir,
            local_checkpoint_dir=local_checkpoint_dir,
            publish_staging_dir=delta_cfg.publish_staging_dir,
            max_file_size_in_gb=delta_cfg.max_file_size_in_gb,
            gcs_download_workers=delta_cfg.gcs_download_workers,
            publish_num_workers=delta_cfg.publish_num_workers,
            checkpoint_load_format=delta_cfg.checkpoint_load_format,
            multi_thread_safetensors_max_workers=delta_cfg.multi_thread_safetensors_max_workers,
            override_existing_receiver=not ie_cfg.run_engines_locally,
        )

    @staticmethod
    def create_sender(
        init_info: DeltaInitInfo,
        inference_client: "RemoteInferenceClient",
    ) -> DeltaWeightTransferSender:
        return DeltaWeightTransferSender(init_info=init_info, inference_client=inference_client)

    @staticmethod
    def get_vllm_transfer_engine() -> type:
        from skyrl.backends.skyrl_train.weight_sync.delta_engine import (
            DeltaWeightTransferEngine,
        )

        return DeltaWeightTransferEngine
