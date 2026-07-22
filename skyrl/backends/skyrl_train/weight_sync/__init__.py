"""Weight synchronization abstractions for distributed RL training."""

from typing import Type

from .base import LoraLoadRequest, WeightChunk, WeightUpdateRequest
from .broadcast_strategy import (
    BroadcastInitInfo,
    BroadcastTransferStrategy,
    BroadcastWeightTransferSender,
    BroadcastWeightUpdateRequest,
)
from .cuda_ipc_strategy import (
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    CudaIpcWeightTransferSender,
    CudaIpcWeightUpdateRequest,
)
from .sharded_rdt_strategy import (
    ShardedRdtInitInfo,
    ShardedRdtTransferStrategy,
    ShardedRdtWeightTransferSender,
)
from .transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferSender,
    WeightTransferStrategy,
)
from .weight_extractor import WeightExtractor


def get_transfer_strategy_cls(weight_sync_backend: str, colocate_all: bool) -> Type[WeightTransferStrategy]:
    """Get the appropriate transfer strategy class based on config.

    Uses CUDA IPC when:
    - weight_sync_backend is "nccl"
    - colocate_all is True (training and inference on same nodes)

    Otherwise uses broadcast.

    Args:
        weight_sync_backend: The weight sync backend ("nccl" or other).
        colocate_all: Whether training and inference are colocated on same nodes.

    Returns:
        The strategy class (CudaIpcTransferStrategy or BroadcastTransferStrategy).
    """
    strategy = get_transfer_strategy(weight_sync_backend, colocate_all)
    if strategy == "sharded_rdt":
        # Ensure the engine is registered in vLLM's factory driver-side too
        # (idempotent; no-op without vLLM). Worker-side registration happens via
        # the worker-extension import.
        from . import rdt_vllm_register

        rdt_vllm_register.ensure_registered()
        return ShardedRdtTransferStrategy
    if strategy == "ipc":
        return CudaIpcTransferStrategy
    return BroadcastTransferStrategy


def get_transfer_strategy(weight_sync_backend: str, colocate_all: bool) -> str:
    """Get the appropriate transfer strategy string based on config."""
    if weight_sync_backend in ("sharded_rdt", "rdt"):
        return "sharded_rdt"
    if weight_sync_backend == "nccl" and colocate_all:
        return "ipc"
    return "nccl"


__all__ = [
    "WeightChunk",
    "WeightExtractor",
    "WeightUpdateRequest",
    "LoraLoadRequest",
    "BroadcastWeightUpdateRequest",
    "CudaIpcWeightUpdateRequest",
    "WeightTransferStrategy",
    "WeightTransferSender",
    "WeightSyncInitInfo",
    "BroadcastInitInfo",
    "CudaIpcInitInfo",
    "BroadcastTransferStrategy",
    "BroadcastWeightTransferSender",
    "CudaIpcTransferStrategy",
    "CudaIpcWeightTransferSender",
    "ShardedRdtInitInfo",
    "ShardedRdtTransferStrategy",
    "ShardedRdtWeightTransferSender",
    "get_transfer_strategy",
    "get_transfer_strategy_cls",
]
