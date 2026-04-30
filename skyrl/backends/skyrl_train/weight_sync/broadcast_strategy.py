"""Broadcast-based weight transfer strategy using torch.distributed.

This module implements the broadcast transfer strategy for synchronizing model weights
from training workers to inference engines using NCCL/Gloo broadcast operations.
"""

import asyncio
import socket
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, List, Optional, Tuple

if TYPE_CHECKING:
    from skyrl.train.config.config import InferenceEngineConfig

import ray
import torch

from skyrl.backends.skyrl_train.distributed.utils import init_custom_process_group
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import (
    InferenceEngineClient,
)
from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk, WeightUpdateRequest
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferReceiver,
    WeightTransferSender,
    WeightTransferStrategy,
)
from skyrl.env_vars import _SKYRL_USE_NEW_INFERENCE
from skyrl.train.utils.utils import get_tcp_url


@dataclass
class BroadcastInitInfo(WeightSyncInitInfo):
    """Initialization info for broadcast-based weight transfer."""

    master_addr: str
    master_port: int
    rank_offset: int
    world_size: int
    group_name: str
    backend: str
    model_dtype_str: str

    @staticmethod
    def strategy_type() -> type:
        """Return the strategy class for this init info type."""
        return BroadcastTransferStrategy

    def for_engine(self, engine_index: int, tp_size: int, pp_size: int, dp_size: int) -> "BroadcastInitInfo":
        """Return init_info with rank_offset adjusted for this engine.

        Args:
            engine_index: Index of the engine (0-based).
            tp_size: Tensor parallel size of the engine.
            pp_size: Pipeline parallel size of the engine.
            dp_size: Data parallel size of the engine.

        Returns:
            BroadcastInitInfo with adjusted rank_offset.
        """
        cumulative_offset = engine_index * tp_size * pp_size * dp_size
        return replace(self, rank_offset=self.rank_offset + cumulative_offset)

    # TODO (Aaron): native weight sync only needs the following params:
    #     master_address, master_port, rank_offset, world_size
    # so we need a new method (to_api_payload) to return the payload for the native weight sync.
    # Also we need a new method (for_servers) to update the rank_offset for the native weight
    # sync, since this is done automatically in the legacy weight sync.

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
        inference_client: InferenceEngineClient,
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
                "shapes". When provided on the vLLM native path, avoids materializing
                all chunks to collect metadata. Ignored on legacy path.
        """
        if _SKYRL_USE_NEW_INFERENCE:
            await self._send_chunks_vllm_native(chunks, weight_metadata)
        else:
            await self._send_chunks_legacy(chunks)

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

        if torch.distributed.get_rank() == 0:
            from vllm.distributed.weight_transfer.nccl_engine import (
                NCCLWeightTransferEngine,
            )

            update_info = {**weight_metadata, "packed": True}
            update_task = asyncio.create_task(self._inference_client.update_named_weights(update_info))

            # Run in thread so the HTTP update_task can progress concurrently
            await asyncio.to_thread(
                NCCLWeightTransferEngine.trainer_send_weights,
                iterator=weight_iterator(),
                trainer_args={"group": self._model_update_group, "packed": True},
            )
            await update_task
        else:
            # Non-rank-0 still needs to participate in the all-gather
            for _ in weight_iterator():
                pass

        torch.distributed.barrier()

    async def _send_chunks_legacy(self, chunks: Iterable[WeightChunk]) -> None:
        """Per-chunk packed broadcast (legacy path).

        Packs all tensors in each chunk into a single contiguous buffer and
        broadcasts it in one NCCL operation, reducing per-tensor broadcast overhead.
        """
        rank = torch.distributed.get_rank()

        # Rank 0 must have a process group to broadcast to inference engines
        if rank == 0:
            assert self._model_update_group is not None, "Rank 0 must have model_update_group"

        # All ranks iterate through chunks (weight extraction may involve collective ops)
        for chunk in chunks:
            # Only rank 0 sends request to inference engines
            sizes = [t.numel() for t in chunk.tensors]

            if rank == 0:
                from skyrl.train.utils.utils import str_to_torch_dtype

                dtype = str_to_torch_dtype(self._init_info.model_dtype_str)
                device = torch.cuda.current_device()

                total_numel = sum(sizes)
                packed_tensor = torch.empty(total_numel, device=device, dtype=dtype, requires_grad=False)
                offset = 0
                for tensor, size in zip(chunk.tensors, sizes):
                    packed_tensor[offset : offset + size].copy_(tensor.detach().view(-1))
                    offset += size

                request = BroadcastWeightUpdateRequest(
                    names=chunk.names,
                    dtypes=[self._init_info.model_dtype_str] * len(chunk),
                    shapes=chunk.shapes,
                    sizes=sizes,
                )
                update_weight_task = asyncio.create_task(self._inference_client.update_named_weights(request))

                def broadcast_packed(t, group):
                    torch.distributed.broadcast(t.data, 0, group=group)

                await asyncio.to_thread(broadcast_packed, packed_tensor, self._model_update_group)
                await update_weight_task

            torch.distributed.barrier()

    def teardown(self) -> None:
        """Destroy the process group used for weight transfer."""
        if self._model_update_group is not None and isinstance(
            self._model_update_group, torch.distributed.ProcessGroup
        ):
            torch.distributed.destroy_process_group(self._model_update_group)
        self._model_update_group = None


class BroadcastWeightTransferReceiver(WeightTransferReceiver):
    """Receives weights via torch.distributed.broadcast.

    Allocates tensors locally and receives data via broadcast from training workers.
    """

    def __init__(
        self,
        model_dtype: torch.dtype,
        model_update_group: torch.distributed.ProcessGroup,
    ) -> None:
        """Initialize the broadcast receiver.

        Args:
            model_dtype: Expected dtype for received tensors.
            model_update_group: Process group for broadcast operations.
        """
        self._model_dtype = model_dtype
        self._model_update_group = model_update_group

    def receive_weights(self, request: BroadcastWeightUpdateRequest) -> Iterator[Tuple[str, torch.Tensor]]:
        """Receive weights via broadcast.

        When request.sizes is set (packed mode), receives a single contiguous
        buffer and unpacks into individual tensors. Otherwise falls back to
        per-tensor broadcast for backward compatibility.

        Args:
            request: Broadcast weight update request with names, dtypes, shapes.

        Yields:
            Tuples of (parameter_name, tensor) for each weight.
        """
        from skyrl.train.utils.utils import str_to_torch_dtype

        if request.sizes is not None:
            if not request.sizes:
                return

            assert len(request.sizes) == len(request), "sizes must match number of parameters"
            dtype = str_to_torch_dtype(request.dtypes[0])
            assert dtype == self._model_dtype, f"dtype mismatch: request {dtype}, model {self._model_dtype}"

            total_numel = sum(request.sizes)
            packed = torch.empty(total_numel, dtype=dtype, device="cuda")
            torch.distributed.broadcast(packed, 0, group=self._model_update_group)

            offset = 0
            for name, shape, size in zip(request.names, request.shapes, request.sizes):
                yield name, packed[offset : offset + size].view(*shape)
                offset += size
        else:
            for name, dtype_str, shape in zip(request.names, request.dtypes, request.shapes):
                dtype = str_to_torch_dtype(dtype_str)
                assert dtype == self._model_dtype, f"dtype mismatch: request {dtype}, model {self._model_dtype}"
                weight = torch.empty(shape, dtype=dtype, device="cuda")
                torch.distributed.broadcast(weight, 0, group=self._model_update_group)
                yield name, weight

    def teardown(self) -> None:
        """Destroy the process group used for weight transfer."""
        torch.distributed.destroy_process_group(self._model_update_group)


class BroadcastTransferStrategy(WeightTransferStrategy):
    """Factory for broadcast-based weight transfer.

    This strategy uses NCCL/Gloo broadcast operations to transfer weights from
    training workers to inference engines.

    All methods are static - no instance state needed.
    """

    @staticmethod
    def create_init_info(
        ie_cfg: "InferenceEngineConfig", inference_world_size: Optional[int] = None
    ) -> BroadcastInitInfo:
        """Create init info with all config-derived args.

        Args:
            ie_cfg: InferenceEngineConfig containing inference engine settings.
            inference_world_size: Total number of inference workers (from client.get_world_size()).
                If provided, uses this instead of calculating from config.
                This is the preferred approach for HTTP inference path.

        Returns:
            BroadcastInitInfo containing all args needed for sender/receiver creation.
        """

        if _SKYRL_USE_NEW_INFERENCE:
            # New inference path: use world_size from servers
            if inference_world_size is None:
                raise ValueError("inference_world_size must be provided when using new inference path")
            world_size = inference_world_size + 1  # +1 for trainer rank 0
        else:
            # Legacy path: calculate from config
            num_inference_engines = ie_cfg.num_engines
            tensor_parallel_size = ie_cfg.tensor_parallel_size
            pipeline_parallel_size = ie_cfg.pipeline_parallel_size
            data_parallel_size = ie_cfg.data_parallel_size
            world_size = num_inference_engines * tensor_parallel_size * pipeline_parallel_size * data_parallel_size + 1

        master_addr = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        return BroadcastInitInfo(
            master_addr=master_addr,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
            group_name="skyrl",
            backend=ie_cfg.weight_sync_backend,
            model_dtype_str=ie_cfg.model_dtype,
            override_existing_receiver=ie_cfg.override_existing_update_group == "enable",
        )

    @staticmethod
    def create_sender(
        init_info: BroadcastInitInfo,
        inference_client: InferenceEngineClient,
    ) -> BroadcastWeightTransferSender:
        """Create a broadcast sender.

        When _SKYRL_USE_NEW_INFERENCE, uses vLLM's NCCLWeightTransferEngine.trainer_init
        on rank 0. Otherwise uses init_custom_process_group for legacy path.

        Args:
            init_info: BroadcastInitInfo from create_init_info.
            inference_client: Client for coordinating with inference engines.
        """
        rank = torch.distributed.get_rank()
        model_update_group = None

        if rank == 0:
            if _SKYRL_USE_NEW_INFERENCE:
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
            else:
                model_update_group = init_custom_process_group(
                    backend=init_info.backend,
                    init_method=get_tcp_url(init_info.master_addr, init_info.master_port),
                    world_size=init_info.world_size,
                    rank=0,
                    group_name=init_info.group_name,
                )

        return BroadcastWeightTransferSender(
            init_info=init_info,
            model_update_group=model_update_group,
            inference_client=inference_client,
        )

    @staticmethod
    def create_receiver(init_info: BroadcastInitInfo) -> BroadcastWeightTransferReceiver:
        """Create a broadcast receiver.

        Sets up the process group and returns a configured receiver.

        Args:
            init_info: BroadcastInitInfo from the sender.

        Returns:
            A configured BroadcastWeightTransferReceiver instance.
        """
        from skyrl.train.utils.utils import str_to_torch_dtype

        # Setup process group (receiver rank = local rank + rank_offset)
        rank = torch.distributed.get_rank() + init_info.rank_offset
        model_update_group = init_custom_process_group(
            backend=init_info.backend,
            init_method=get_tcp_url(init_info.master_addr, init_info.master_port),
            world_size=init_info.world_size,
            rank=rank,
            group_name=init_info.group_name,
        )

        model_dtype = str_to_torch_dtype(init_info.model_dtype_str)
        return BroadcastWeightTransferReceiver(
            model_dtype=model_dtype,
            model_update_group=model_update_group,
        )
