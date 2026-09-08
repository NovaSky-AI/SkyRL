"""Broadcast-based weight transfer strategy using torch.distributed.

This module implements the broadcast transfer strategy for synchronizing model weights
from training workers to inference engines using NCCL/Gloo broadcast operations.
"""

import asyncio
import socket
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    from skyrl.backends.skyrl_train.inference_servers.remote_inference_client import (
        RemoteInferenceClient,
    )
    from skyrl.train.config.config import InferenceEngineConfig

import ray
import torch

from skyrl.backends.skyrl_train.weight_sync.base import (
    WeightChunk,
    WeightUpdateRequest,
    get_weight_chunk_metadata,
)
from skyrl.backends.skyrl_train.weight_sync.nccl_trainer_send import (
    nccl_trainer_init,
    nccl_trainer_send_weights,
)
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferSender,
    WeightTransferStrategy,
)


@dataclass
class BroadcastInitInfo(WeightSyncInitInfo):
    """Initialization info for broadcast-based weight transfer."""

    master_addr: str
    master_port: int
    rank_offset: int
    world_size: int
    packed: bool = True
    """Whether the transfer is packed. As of vLLM 0.28.0 this is an init-time
    wire param: the worker records it from ``NCCLWeightTransferInitInfo`` during
    ``/init_weight_transfer_engine`` and ``receive_weights`` reads it from there,
    so it can no longer ride the per-round update info. It must match the
    ``packed`` the sender passes to ``nccl_trainer_send_weights``, or the two
    sides split the stream differently and the broadcast hangs in NCCL."""

    def for_servers(self, world_size_per_server: int, num_servers: int, dp_size: int = 1) -> List["BroadcastInitInfo"]:
        """Return one BroadcastInitInfo per server with rank_offset for each.

        Used when calling init_weight_update_communicator on the new inference path:
        expand the single init_info into a list (one per server), then pass
        [x.to_api_payload() for x in server_infos] to the client.

        server_urls are ordered as [engine0_dp0, engine0_dp1, ..., engine1_dp0, ...].
        All DP servers within one deployment share the same rank_offset because
        vLLM's init_transfer_engine already accounts for dp_rank internally.
        The offset only advances at deployment (num_engines) boundaries.

        Args:
            world_size_per_server: Number of workers per server (same for all servers).
            num_servers: Total number of servers (num_engines * dp_size).
            dp_size: Data parallel size. Servers are grouped into deployments
                of dp_size servers each.

        Returns:
            List of BroadcastInitInfo, one per server, with cumulative rank_offset.
        """
        result: List[BroadcastInitInfo] = []
        rank_offset = self.rank_offset
        for i in range(num_servers):
            result.append(replace(self, rank_offset=rank_offset))
            # Advance rank_offset only at deployment boundaries (every dp_size servers)
            if (i + 1) % dp_size == 0:
                rank_offset += world_size_per_server
        return result

    def to_api_payload(self) -> Dict[str, Any]:
        """Return JSON-serializable payload for the /init_weight_transfer_engine endpoint."""
        return {
            "master_address": self.master_addr,
            "master_port": self.master_port,
            "rank_offset": self.rank_offset,
            "world_size": self.world_size,
            "packed": self.packed,
        }


@dataclass
class BroadcastWeightUpdateRequest(WeightUpdateRequest):
    """Request for broadcast-based weight transfer.

    When sizes is provided, tensors are packed into a single contiguous buffer
    and broadcast as one NCCL operation per chunk. The receiver uses sizes to unpack.
    When sizes is None, falls back to per-tensor broadcast (backward compatible).
    """

    sizes: Optional[List[int]] = None


class BroadcastWeightTransferSender(WeightTransferSender):
    """Sends weights via torch.distributed.broadcast or vLLM NCCL (new inference path).

    When using new inference, uses the vendored ``nccl_trainer_send_weights``
    (see ``nccl_trainer_send.py``) with batched update_weights. Otherwise uses
    per-chunk HTTP + torch.distributed.broadcast.
    """

    def __init__(
        self,
        init_info: BroadcastInitInfo,
        model_update_group: Optional[Any],
        inference_client: "RemoteInferenceClient",
    ) -> None:
        """Initialize the broadcast sender.

        Args:
            init_info: BroadcastInitInfo containing all config-derived args.
            model_update_group: Communication group for weight transfer. Either a
                torch.distributed.ProcessGroup (legacy) or a vLLM NCCL
                communicator (new path). None on non-rank-0 workers.
            inference_client: Client for coordinating with inference engines.
        """
        self._init_info = init_info
        self._model_update_group = model_update_group
        self._inference_client = inference_client

    async def send_chunks(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
        derive_metadata_from_chunks: bool = False,
        **kwargs,
    ) -> None:
        """Send chunks via broadcast or vLLM native NCCL.

        Args:
            chunks: Iterable of WeightChunk objects to send.
            weight_metadata: Complete metadata for the batched update path.
            derive_metadata_from_chunks: Send each chunk with derived metadata.
        """
        if derive_metadata_from_chunks:
            if weight_metadata is not None:
                raise ValueError("weight_metadata must be omitted when deriving metadata from chunks")
            await self._send_serialized_fp8_chunks_vllm_native(chunks)
        else:
            await self._send_chunks_vllm_native(chunks, weight_metadata)

    async def _send_chunks_vllm_native(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]],
    ) -> None:
        """Batched path: one update_weights call + nccl_trainer_send_weights.

        All ranks must evaluate the chunks iterator (extract_weights uses
        collective all-gather internally). Only rank 0 sends the gathered
        tensors to vLLM via the NCCL weight transfer engine.
        """
        if weight_metadata is None:
            raise ValueError("weight_metadata is required unless derive_metadata_from_chunks=true")

        def weight_iterator() -> Iterator[Tuple[str, torch.Tensor]]:
            for chunk in chunks:
                yield from zip(chunk.names, chunk.tensors)

        # Route via the skyrl wrap (start_weight_update + update_weights_nccl
        # + finish_weight_update) rather than vLLM's native /update_weights so
        # the receive is wrapped with set_current_vllm_config. Matches how
        # CUDA IPC already routes through skyrl's wrap.
        # TODO: switch back to update_named_weights once the upstream vLLM
        # patch lands (vllm-project/vllm weight-sync-fix).
        # https://github.com/vllm-project/vllm/pull/42577
        if torch.distributed.get_rank() == 0:
            await self._inference_client.start_weight_update(is_checkpoint_format=True)

            # vLLM 0.28.0 dropped `packed` (and the buffer geometry) from
            # NCCLWeightTransferUpdateInfo -- it is agreed once at init instead,
            # via BroadcastInitInfo.packed. Sending it here is now a TypeError.
            update_info = dict(weight_metadata)
            update_task = asyncio.create_task(self._inference_client.update_weights_nccl(update_info))

            # Run in a thread so the HTTP update task can progress concurrently.
            await asyncio.to_thread(
                nccl_trainer_send_weights,
                weight_iterator(),
                self._model_update_group,
                packed=self._init_info.packed,
            )
            await update_task

            await self._inference_client.finish_weight_update()
        else:
            # Non-rank-0 still needs to participate in extractor collectives.
            for _ in weight_iterator():
                pass

        torch.distributed.barrier()

    async def _send_serialized_fp8_chunks_vllm_native(
        self,
        chunks: Iterable[WeightChunk],
    ) -> None:
        """Send lazy mixed-dtype serialized-FP8 chunks through vLLM NCCL."""
        if torch.distributed.get_rank() == 0:
            await self._inference_client.start_weight_update(is_checkpoint_format=True)

        for chunk in chunks:
            if torch.distributed.get_rank() == 0:
                await self._send_chunk_vllm_native(chunk)

        if torch.distributed.get_rank() == 0:
            await self._inference_client.finish_weight_update()

        torch.distributed.barrier()

    async def _send_chunk_vllm_native(self, chunk: WeightChunk) -> None:
        """Send one logical chunk as its own NCCL update round.

        Same wire protocol as the batched path (vendored
        ``nccl_trainer_send_weights`` + ``BroadcastInitInfo.packed``), just one
        round per chunk because serialized-FP8 names/shapes are only known once
        the chunk is built. The update info carries only names/dtype_names/shapes:
        vLLM 0.28.0 rejects ``packed`` there (it is fixed at init). The packed
        producer linearizes by bytes, so mixed fp8/fp32/bf16 tensors in one
        round are fine.
        """
        update_info = get_weight_chunk_metadata(chunk)
        update_task = asyncio.create_task(self._inference_client.update_weights_nccl(update_info))

        # Let the receiver enter its collective while the trainer broadcasts.
        await asyncio.to_thread(
            nccl_trainer_send_weights,
            iter(zip(chunk.names, chunk.tensors)),
            self._model_update_group,
            packed=self._init_info.packed,
        )
        await update_task

    def teardown(self) -> None:
        """Destroy the process group used for weight transfer."""
        if self._model_update_group is not None and isinstance(
            self._model_update_group, torch.distributed.ProcessGroup
        ):
            torch.distributed.destroy_process_group(self._model_update_group)
        self._model_update_group = None


class BroadcastTransferStrategy(WeightTransferStrategy):
    """Factory for broadcast-based weight transfer.

    This strategy uses NCCL/Gloo broadcast operations to transfer weights from
    training workers to inference engines.

    All methods are static - no instance state needed.
    """

    @staticmethod
    def create_init_info(
        ie_cfg: "InferenceEngineConfig",
        inference_world_size: int,
        base_model_path: Optional[str] = None,
    ) -> BroadcastInitInfo:
        """Create init info with all config-derived args.

        Args:
            ie_cfg: InferenceEngineConfig containing inference engine settings.
            inference_world_size: Total number of inference workers (from client.get_world_size()).

        Returns:
            BroadcastInitInfo containing all args needed for sender/receiver creation.
        """
        # Use world_size reported by the inference servers (+1 for trainer rank 0).
        world_size = inference_world_size + 1

        master_addr = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        return BroadcastInitInfo(
            master_addr=master_addr,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
            override_existing_receiver=not ie_cfg.run_engines_locally,
        )

    @staticmethod
    def create_sender(
        init_info: BroadcastInitInfo,
        inference_client: "RemoteInferenceClient",
        weight_extractor: Optional[Any] = None,
    ) -> BroadcastWeightTransferSender:
        """Create a broadcast sender.

        On rank 0, joins the weight-transfer group via ``nccl_trainer_init``
        (vLLM's ``nccl_common.trainer_init``). Other ranks hold no communicator.

        Args:
            init_info: BroadcastInitInfo from create_init_info.
            inference_client: Client for coordinating with inference engines.
        """
        rank = torch.distributed.get_rank()
        model_update_group = None

        if rank == 0:
            model_update_group = nccl_trainer_init(
                dict(
                    master_address=init_info.master_addr,
                    master_port=init_info.master_port,
                    world_size=init_info.world_size,
                )
            )

        return BroadcastWeightTransferSender(
            init_info=init_info,
            model_update_group=model_update_group,
            inference_client=inference_client,
        )

    @staticmethod
    def get_vllm_transfer_engine() -> type:
        """Return the vLLM weight-transfer engine class for this strategy (NCCL).

        Reference for the receive side: the inference servers drive this engine
        natively. Currently unused on the sender side (we route through the
        SkyRL ``/collective_rpc`` wrap), kept as the canonical mapping.
        """
        from vllm.distributed.weight_transfer.nccl_engine import (
            NCCLWeightTransferEngine,
        )

        return NCCLWeightTransferEngine
