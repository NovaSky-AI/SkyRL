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

from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk, WeightUpdateRequest
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

    When using new inference, uses vLLM's trainer_send_weights with batched
    update_weights. Otherwise uses per-chunk HTTP + torch.distributed.broadcast.
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
    ) -> None:
        """Send chunks via broadcast or vLLM native NCCL.

        Args:
            chunks: Iterable of WeightChunk objects to send.
            weight_metadata: Pre-computed metadata dict with "names", "dtype_names",
                "shapes". Avoids materializing all chunks to collect metadata.
        """
        await self._send_chunks_vllm_native(chunks, weight_metadata)

    async def _send_chunks_vllm_native(
        self,
        chunks: Iterable[WeightChunk],
        weight_metadata: Optional[Dict[str, list]] = None,
    ) -> None:
        """Batched path: one update_weights call + trainer_send_weights (vLLM native).

        All ranks must evaluate the chunks iterator (extract_weights uses
        collective all-gather internally). Only rank 0 sends the gathered
        tensors to vLLM via the NCCL weight transfer engine.
        """
        if weight_metadata is None:
            raise ValueError(
                "weight_metadata is required for vLLM native path. "
                "Call weight_extractor.get_weight_metadata() and pass it to send_chunks."
            )

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
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLWeightTransferEngine,
            )

            await self._inference_client.start_weight_update(is_checkpoint_format=True)

            update_info = {**weight_metadata, "packed": True}
            update_task = asyncio.create_task(self._inference_client.update_weights_nccl(update_info))

            # Run in thread so the HTTP update_task can progress concurrently
            await asyncio.to_thread(
                NCCLWeightTransferEngine.trainer_send_weights,
                iterator=weight_iterator(),
                trainer_args={"group": self._model_update_group, "packed": True},
            )
            await update_task

            await self._inference_client.finish_weight_update()
        else:
            # Non-rank-0 still needs to participate in the all-gather
            for _ in weight_iterator():
                pass

        torch.distributed.barrier()

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
    def create_init_info(ie_cfg: "InferenceEngineConfig", inference_world_size: int) -> BroadcastInitInfo:
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
    ) -> BroadcastWeightTransferSender:
        """Create a broadcast sender.

        On rank 0, uses vLLM's NCCLWeightTransferEngine.trainer_init to join the
        weight-transfer group. Other ranks do not hold a communicator.

        Args:
            init_info: BroadcastInitInfo from create_init_info.
            inference_client: Client for coordinating with inference engines.
        """
        rank = torch.distributed.get_rank()
        model_update_group = None

        if rank == 0:
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLWeightTransferEngine,
            )

            model_update_group = NCCLWeightTransferEngine.trainer_init(
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
