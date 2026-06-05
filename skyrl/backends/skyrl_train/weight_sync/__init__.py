"""Weight synchronization abstractions for distributed RL training."""

from typing import Type

# Importing this registers the ``sharded_rdt`` engine into vLLM's factory and
# relaxes WeightTransferConfig.backend to accept it. No-op (swallows ImportError)
# on platforms without the vLLM wheel, so it is safe to import on CPU. Covers the
# trainer-driver process; workers are covered via new_inference_worker_wrap.py.
from . import rdt_vllm_register  # noqa: F401
from .base import LoraLoadRequest, WeightChunk, WeightUpdateRequest
from .broadcast_strategy import (
    BroadcastInitInfo,
    BroadcastTransferStrategy,
    BroadcastWeightTransferReceiver,
    BroadcastWeightTransferSender,
    BroadcastWeightUpdateRequest,
)
from .cuda_ipc_strategy import (
    CudaIpcInitInfo,
    CudaIpcTransferStrategy,
    CudaIpcWeightTransferReceiver,
    CudaIpcWeightTransferSender,
    CudaIpcWeightUpdateRequest,
)
from .sharded_rdt_strategy import (
    RDT_TRAINER_ACTOR_NAME,
    RdtProducerMixin,
    ShardedRdtInitInfo,
    ShardedRdtTransferStrategy,
    ShardedRdtWeightTransferSender,
)
from .transfer_strategy import (
    WeightSyncInitInfo,
    WeightTransferReceiver,
    WeightTransferSender,
    WeightTransferStrategy,
)
from .weight_extractor import WeightExtractor
from .weight_loader import WeightLoader


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
    if strategy == "ipc":
        return CudaIpcTransferStrategy
    if strategy == "sharded_rdt":
        return ShardedRdtTransferStrategy
    return BroadcastTransferStrategy


def get_transfer_strategy(weight_sync_backend: str, colocate_all: bool) -> str:
    """Get the appropriate transfer strategy string based on config.

    This string is BOTH the SkyRL strategy selector and the vLLM
    ``WeightTransferConfig.backend`` value. ``sharded_rdt`` is a pull-based
    NIXL backend (non-colocated only — workers pull from a named trainer actor).
    """
    if weight_sync_backend == "sharded_rdt":
        return "sharded_rdt"
    if weight_sync_backend == "nccl" and colocate_all:
        return "ipc"
    return "nccl"


__all__ = [
    "WeightChunk",
    "WeightExtractor",
    "WeightLoader",
    "WeightUpdateRequest",
    "LoraLoadRequest",
    "BroadcastWeightUpdateRequest",
    "CudaIpcWeightUpdateRequest",
    "WeightTransferStrategy",
    "WeightTransferSender",
    "WeightTransferReceiver",
    "WeightSyncInitInfo",
    "BroadcastInitInfo",
    "CudaIpcInitInfo",
    "BroadcastTransferStrategy",
    "BroadcastWeightTransferSender",
    "BroadcastWeightTransferReceiver",
    "CudaIpcTransferStrategy",
    "CudaIpcWeightTransferSender",
    "CudaIpcWeightTransferReceiver",
    "ShardedRdtInitInfo",
    "ShardedRdtTransferStrategy",
    "ShardedRdtWeightTransferSender",
    "RdtProducerMixin",
    "RDT_TRAINER_ACTOR_NAME",
    "get_transfer_strategy",
    "get_transfer_strategy_cls",
]
