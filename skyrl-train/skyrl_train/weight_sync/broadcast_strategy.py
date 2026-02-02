"""Broadcast-based weight transfer strategy using torch.distributed.

This module implements the broadcast transfer strategy for synchronizing model weights
from training workers to inference engines using NCCL/Gloo broadcast operations.
"""

import asyncio
import socket
from dataclasses import dataclass, replace
from typing import Iterable, Iterator, Optional, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from omegaconf import DictConfig
    from skyrl_train.config import SkyRLConfig

import ray
import torch

from skyrl_train.distributed.utils import init_custom_process_group
from skyrl_train.env_vars import _SKYRL_USE_NEW_INFERENCE
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.utils import get_tcp_url
from skyrl_train.weight_sync.base import WeightChunk, WeightUpdateRequest
from skyrl_train.weight_sync.transfer_strategy import (
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


@dataclass
class BroadcastWeightUpdateRequest(WeightUpdateRequest):
    """Request for broadcast-based weight transfer.

    Contains only metadata - actual tensor data is sent via torch.distributed.broadcast.
    """

    pass


class BroadcastWeightTransferSender(WeightTransferSender):
    """Sends weights via torch.distributed.broadcast.

    The sender broadcasts tensors from rank 0 to all other ranks in the
    model update group, while coordinating with inference engines via RPC.
    """

    def __init__(
        self,
        init_info: BroadcastInitInfo,
        model_update_group: Optional[torch.distributed.ProcessGroup],
        inference_client: InferenceEngineClient,
    ) -> None:
        """Initialize the broadcast sender.

        Args:
            init_info: BroadcastInitInfo containing all config-derived args.
            model_update_group: Process group for broadcast operations (None on non-rank-0 training workers).
            inference_client: Client for coordinating with inference engines.
        """
        self._init_info = init_info
        self._model_update_group = model_update_group
        self._inference_client = inference_client

    async def send_chunks(self, chunks: Iterable[WeightChunk]) -> None:
        """Send chunks via broadcast.

        Each chunk should contain exactly one parameter for broadcast strategy.
        All training ranks iterate through chunks (weight extraction may involve
        collective ops), but only rank 0 broadcasts to inference engines via the
        model_update_group.

        Args:
            chunks: Iterable of WeightChunk objects to send.
        """
        rank = torch.distributed.get_rank()

        # Rank 0 must have a process group to broadcast to inference engines
        if rank == 0:
            assert self._model_update_group is not None, "Rank 0 must have model_update_group"

        # All ranks iterate through chunks (weight extraction may involve collective ops)
        for chunk in chunks:
            assert len(chunk) == 1, f"Broadcast strategy expects single-parameter chunks, got {len(chunk)}"

            name = chunk.names[0]
            tensor = chunk.tensors[0]
            shape = chunk.shapes[0]

            # Only rank 0 sends request to inference engines
            if rank == 0:
                request = BroadcastWeightUpdateRequest(
                    names=[name],
                    dtypes=[self._init_info.model_dtype_str],
                    shapes=[shape],
                )
                update_weight_task = asyncio.create_task(self._inference_client.update_named_weights(request))

            # Broadcast tensor from rank 0 to inference engines (no-op on other training ranks)
            def broadcast_tensor(t: torch.Tensor) -> None:
                if rank == 0 and self._model_update_group is not None:
                    torch.distributed.broadcast(t.data, 0, group=self._model_update_group)

            await asyncio.to_thread(broadcast_tensor, tensor)

            if rank == 0:
                await update_weight_task

            torch.distributed.barrier()

    def teardown(self) -> None:
        """Destroy the process group used for weight transfer.

        TODO: Integrate with training workers to call this during shutdown.
        """
        if self._model_update_group is not None:
            torch.distributed.destroy_process_group(self._model_update_group)


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
        from skyrl_train.utils import str_to_torch_dtype

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
        cfg: "Union[SkyRLConfig, DictConfig]", inference_world_size: Optional[int] = None
    ) -> BroadcastInitInfo:
        """Create init info with all config-derived args.

        Args:
            cfg: Configuration object containing generator settings.
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
            num_inference_engines = cfg.generator.num_inference_engines
            tensor_parallel_size = cfg.generator.inference_engine_tensor_parallel_size
            pipeline_parallel_size = cfg.generator.inference_engine_pipeline_parallel_size
            data_parallel_size = cfg.generator.inference_engine_data_parallel_size
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
            backend=cfg.generator.weight_sync_backend,
            model_dtype_str=cfg.generator.model_dtype,
            override_existing_receiver=cfg.generator.override_existing_update_group == "enable",
        )

    @staticmethod
    def create_sender(
        init_info: BroadcastInitInfo,
        inference_client: InferenceEngineClient,
    ) -> BroadcastWeightTransferSender:
        """Create a broadcast sender.

        Sets up the process group on rank 0 only (other training ranks don't join).

        Args:
            init_info: BroadcastInitInfo containing config-derived args.
            inference_client: Client for coordinating with inference engines.

        Returns:
            A configured BroadcastWeightTransferSender instance.
        """
        # Only rank 0 joins the model_update_group (with inference engines)
        # Other training ranks don't participate in the process group
        model_update_group = None
        if torch.distributed.get_rank() == 0:
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
        from skyrl_train.utils import str_to_torch_dtype

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
