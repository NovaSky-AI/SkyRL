"""Weight extractor interface for extracting weights from training backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterator, List, Literal

import torch

from skyrl.backends.skyrl_train.distributed.dispatch import MeshRank

from .base import WeightChunk

MeshDim = Literal["dp", "sp", "tp", "pp", "cp", "ep", "etp"]


@dataclass(frozen=True)
class ExtractorShardInfo:
    """Describes how a backend extractor's chunk stream is distributed.

    ``replicate_world_size`` and ``source_index_in_replicate_world`` are used
    by disk-delta publishing to split identical streams across equivalent
    source ranks. ``rank`` is only used for artifact naming and logging.
    """

    is_source_rank: bool
    replicate: List[MeshDim]
    split: List[MeshDim]
    mesh_rank: MeshRank
    replicate_world_size: int
    source_index_in_replicate_world: int
    rank: int


class WeightExtractor(ABC):
    """Extracts weights from training backend models.

    Subclasses implement backend-specific logic to extract model weights,
    handle sharding, and prepare them for transfer to inference engines.
    """

    @abstractmethod
    def extract_weights(self, dtype: torch.dtype) -> Iterator[WeightChunk]:
        """Extract weights from the model as WeightChunk objects.

        Implementations should:
        - Gather sharded weights into full tensors
        - Convert tensors to the specified dtype for inference
        - Ensure tensors are contiguous in memory
        - Optionally group related parameters (e.g., QKV for efficiency)

        Args:
            dtype: Target dtype for inference (e.g., torch.bfloat16, torch.float16)

        Yields:
            WeightChunk objects containing model parameters ready for transfer
        """
        ...

    @abstractmethod
    def get_weight_metadata(self, dtype: torch.dtype) -> Dict[str, List]:
        """Return weight metadata without materializing tensors.

        Args:
            dtype: Target dtype for inference (used for dtype name).

        Returns:
            Dict with keys "names", "dtype_names", "shapes".
        """
        ...

    def get_shard_info(self) -> ExtractorShardInfo:
        """Return backend-agnostic information about this extractor's stream.

        The default preserves the historical behavior: all ranks enter
        extraction, but only global rank 0 publishes deltas.
        """
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
            world_size = torch.distributed.get_world_size()
        else:
            rank = 0
            world_size = 1
        return ExtractorShardInfo(
            is_source_rank=(rank == 0),
            replicate=[],
            split=[],
            mesh_rank=MeshRank(dp=rank, sp=0, tp=0, pp=0, world_size=world_size, dp_size=world_size, pp_size=1),
            replicate_world_size=1,
            source_index_in_replicate_world=0,
            rank=rank,
        )
