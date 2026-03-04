"""Broadcast-based weight transfer strategy using torch.distributed.

This module implements the broadcast transfer strategy for synchronizing model weights
from training workers to inference engines using NCCL/Gloo broadcast operations.
"""

import asyncio
import socket
from dataclasses import dataclass, replace
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from skyrl.train.config.config import InferenceEngineConfig

import ray
import torch

from skyrl.backends.skyrl_train.distributed.utils import init_custom_process_group
from skyrl.backends.skyrl_train.env_vars import _SKYRL_USE_NEW_INFERENCE
from skyrl.backends.skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl.train.utils.utils import get_tcp_url
from skyrl.backends.skyrl_train.weight_sync.base import WeightChunk, WeightUpdateRequest
from skyrl.backends.skyrl_train.weight_sync.transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferStrategy,
    WeightTransferSender,
    WeightTransferReceiver,
)


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

    def for_engine(self, engine_index: int, tp_size: int, pp_size: int) -> "BroadcastInitInfo":
        """Return init_info with rank_offset adjusted for this engine.

        Args:
            engine_index: Index of the engine (0-based).
            tp_size: Tensor parallel size of the engine.
            pp_size: Pipeline parallel size of the engine.

        Returns:
            BroadcastInitInfo with adjusted rank_offset.
        """
        cumulative_offset = engine_index * tp_size * pp_size
        return replace(self, rank_offset=self.rank_offset + cumulative_offset)

    # TODO (Aaron): native weight sync only needs the following params:
    #     master_address, master_port, rank_offset, world_size
    # so we need a new method (to_api_payload) to return the payload for the native weight sync.
    # Also we need a new method (update_rank_offset) to update the rank_offset for the native weight
    # sync, since this is done automatically in the legacy weight sync.

    def update_rank_offset(self, world_size_per_server: List[int]) -> List["BroadcastInitInfo"]:
        """Return one BroadcastInitInfo per server with rank_offset for each.

        Used when calling init_weight_update_communicator on the new inference path:
        expand the single init_info into a list (one per server), then pass
        [x.to_api_payload() for x in server_infos] to the client.

        Args:
            world_size_per_server: Per-server worker counts in server order.

        Returns:
            List of BroadcastInitInfo, one per server, with rank_offset 1, 1+ws[0], etc.
        """
        result: List[BroadcastInitInfo] = []
        rank_offset = self.rank_offset
        for ws in world_size_per_server:
            result.append(replace(self, rank_offset=rank_offset))
            rank_offset += ws
        return result

    def to_api_payload(self) -> Union["BroadcastInitInfo", Dict[str, Any]]:
        """Return payload for init_weight_update_communicator.

        When using new inference (vLLM native APIs), returns a dict with
        master_address, master_port, rank_offset, world_size. Otherwise returns
        self for legacy clients that expect WeightSyncInitInfo.
        """
        if _SKYRL_USE_NEW_INFERENCE:
            return {
                "master_address": self.master_addr,
                "master_port": self.master_port,
                "rank_offset": self.rank_offset,
                "world_size": self.world_size,
            }
        return self


@dataclass
class BroadcastWeightUpdateRequest(WeightUpdateRequest):
    """Request for broadcast-based weight transfer.

    Contains only metadata - actual tensor data is sent via torch.distributed.broadcast.
    """

    pass


class BroadcastWeightTransferSender(WeightTransferSender):
    """Sends weights via torch.distributed.broadcast or vLLM NCCL (new inference path).

    When _vllm_group is set, uses vLLM's trainer_send_weights with batched
    update_weights. Otherwise uses per-chunk HTTP + torch.distributed.broadcast.
    """

    def __init__(
        self,
        init_info: BroadcastInitInfo,
        model_update_group: Optional[torch.distributed.ProcessGroup],
        inference_client: InferenceEngineClient,
        vllm_group: Optional[Any] = None,
    ) -> None:
        """Initialize the broadcast sender.

        Args:
            init_info: BroadcastInitInfo containing all config-derived args.
            model_update_group: Process group for broadcast (legacy path); None when vllm_group set.
            inference_client: Client for coordinating with inference engines.
            vllm_group: vLLM NCCL communicator from NCCLWeightTransferEngine.trainer_init (new path).
        """
        self._init_info = init_info
        self._model_update_group = model_update_group
        self._inference_client = inference_client
        self._vllm_group = vllm_group

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
        from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

        rank = torch.distributed.get_rank()

        if weight_metadata is None:
            raise ValueError(
                "weight_metadata is required for vLLM native path. "
                "Call weight_extractor.get_weight_metadata() and pass it to send_chunks."
            )
        update_info = {**weight_metadata, "packed": True}

        if rank == 0:
            update_task = asyncio.create_task(self._inference_client.update_named_weights(update_info))

        def weight_iterator() -> Iterator[Tuple[str, torch.Tensor]]:
            for chunk in chunks:
                yield from zip(chunk.names, chunk.tensors)

        def do_gather_and_send() -> None:
            if rank == 0 and self._vllm_group is not None:
                NCCLWeightTransferEngine.trainer_send_weights(
                    iterator=weight_iterator(),
                    group=self._vllm_group,
                    packed=True,
                )
            else:
                for _ in weight_iterator():
                    pass

        await asyncio.to_thread(do_gather_and_send)

        if rank == 0:
            await update_task

        torch.distributed.barrier()

    async def _send_chunks_legacy(self, chunks: Iterable[WeightChunk]) -> None:
        """Per-chunk HTTP + torch.distributed.broadcast (legacy path)."""
        rank = torch.distributed.get_rank()

        if rank == 0:
            assert self._model_update_group is not None, "Rank 0 must have model_update_group"

        for chunk in chunks:
            assert len(chunk) == 1, f"Broadcast strategy expects single-parameter chunks, got {len(chunk)}"

            name = chunk.names[0]
            tensor = chunk.tensors[0]
            shape = chunk.shapes[0]

            if rank == 0:
                request = BroadcastWeightUpdateRequest(
                    names=[name],
                    dtypes=[self._init_info.model_dtype_str],
                    shapes=[shape],
                )
                update_weight_task = asyncio.create_task(self._inference_client.update_named_weights(request))

            def broadcast_tensor(t: torch.Tensor) -> None:
                if rank == 0 and self._model_update_group is not None:
                    torch.distributed.broadcast(t.data, 0, group=self._model_update_group)

            await asyncio.to_thread(broadcast_tensor, tensor)

            if rank == 0:
                await update_weight_task

            torch.distributed.barrier()

    def teardown(self) -> None:
        """Destroy the process group used for weight transfer (legacy only)."""
        if self._model_update_group is not None:
            torch.distributed.destroy_process_group(self._model_update_group)
        self._vllm_group = None


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

        Args:
            request: Broadcast weight update request with names, dtypes, shapes.

        Yields:
            Tuples of (parameter_name, tensor) for each weight.
        """
        from skyrl.train.utils.utils import str_to_torch_dtype

        for name, dtype_str, shape in zip(request.names, request.dtypes, request.shapes):
            dtype = str_to_torch_dtype(dtype_str)
            assert dtype == self._model_dtype, f"dtype mismatch: request {dtype}, model {self._model_dtype}"

            # Allocate tensor and receive via broadcast
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
        vllm_group = None

        if _SKYRL_USE_NEW_INFERENCE:
            if rank == 0:
                from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

                vllm_group = NCCLWeightTransferEngine.trainer_init(
                    dict(
                        master_address=init_info.master_addr,
                        master_port=init_info.master_port,
                        world_size=init_info.world_size,
                    )
                )
        else:
            if rank == 0:
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
            vllm_group=vllm_group,
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
