"""Defines dispatch and collect logic for distributed training"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

import ray
from ray import ObjectRef
from ray.actor import ActorHandle

from skyrl.backends.skyrl_train.training_batch import (
    TrainingInputBatch,
    TrainingOutputBatch,
    pad_training_input_batch,
)


@dataclass
class MeshRank:
    """Represents a rank in the device mesh.

    This is a tuple of (DP, SP, TP, PP) ranks.
    """

    dp: int
    sp: int
    tp: int
    pp: int

    world_size: int
    dp_size: int
    pp_size: int

    def is_collection_dp_rank(self) -> bool:
        """Check if this rank is a DP rank to collect from

        This is the rank with (SP=0, TP=0, PP=pp_size-1)

        Note: double check this for ETP > 1 (but this is not a typically used case)
        """
        return self.tp == 0 and self.pp == self.pp_size - 1 and self.sp == 0

    def __str__(self) -> str:
        return f"MeshRank(dp={self.dp}, sp={self.sp}, tp={self.tp}, pp={self.pp}, world_size={self.world_size}, dp_size={self.dp_size}, pp_size={self.pp_size})"

    def __repr__(self) -> str:
        return self.__str__()


@dataclass
class ActorInfo:
    """Actor information for distributed training.

    This includes the actor handle and the rank in the device mesh.
    """

    handle: ActorHandle
    rank: MeshRank


class Dispatch(ABC):
    """Base class for dispatch types

    Dispatch types are responsible for:
    - dispatching method calls to actors handling data sharding if necessary
    - validating arguments for dispatch
    """

    @classmethod
    @abstractmethod
    def dispatch(cls, actor_infos: List[ActorInfo], method: str, *args, **kwargs) -> List[ObjectRef]:
        """Dispatches method calls to the actors with data sharding if necessary."""
        pass

    @classmethod
    @abstractmethod
    def validate_dispatch_args(cls, *args, **kwargs) -> Tuple[Tuple, Dict[str, Any]]:
        """Validate and process arguments for dispatch.

        Returns:
            Tuple of (args, kwargs) to be passed to dispatch
        """
        pass


class MeshDispatch(Dispatch):
    """Mesh dispatch type to dispatch data to a group of actors along the device mesh.

    Supports DP (Data Parallel), SP (Sequence Parallel), TP (Tensor Parallel) and PP (Pipeline Parallel) parallelism.
    The actor method should accept a single argument - the data batch.

    For data dispatch:

    * The input data is chunked into `dp_size` equal chunks, where `dp_size` is the size of data parallelism.
    * Each actor with the same DP rank processes the same data chunk in parallel.

    Example: For a world size of 8, with DP size=2, SP size=2, TP size=2, PP size=1:

    * Data dispatch: The data is chunked into 2 chunks. All actors with DP rank 0 process the first chunk,
      and all actors with DP rank 1 process the second chunk.
    """

    @classmethod
    def dispatch(cls, actor_infos: List[ActorInfo], method: str, data: TrainingInputBatch, **kwargs) -> List[ObjectRef]:
        assert len(actor_infos) > 0, "actor_infos must be a non-empty list"
        object_refs = []
        dp_size = actor_infos[0].rank.dp_size
        assert len(data) % dp_size == 0, "data batch size must be divisible by dp_size, got {} and {}".format(
            len(data), dp_size
        )
        chunk_size = len(data) // dp_size
        data_chunks: List[TrainingInputBatch] = data.chunk(chunk_size)

        # Put each unique chunk in object store ONCE to avoid redundant serialization
        # when the same chunk is sent to multiple workers (e.g., SP/TP replicas)
        chunk_refs: List[ObjectRef] = [ray.put(chunk) for chunk in data_chunks]

        for actor_info in actor_infos:
            # Pass ObjectRef instead of data - workers will fetch from object store
            chunk_ref = chunk_refs[actor_info.rank.dp]
            object_refs.append(getattr(actor_info.handle, method).remote(chunk_ref, **kwargs))
        return object_refs

    @classmethod
    def stage_chunks(
        cls,
        dp_size: int,
        data: TrainingInputBatch,
        mini_batch_boundaries: List[Tuple[int, int]],
    ) -> List[List[ObjectRef]]:
        """Pre-stage mini-batch chunks into the object store.

        Each mini-batch is defined by a ``(start, end)`` index pair from mini_batch_boundaries.
        Mini-batches are individually padded so that their size is divisible by dp_size, using dummy
        entries with ``loss_mask=0`` that do not affect the loss.

        Args:
            dp_size: Number of data-parallel ranks.
            data: Full TrainingInputBatch to slice from.
            mini_batch_boundaries: List of ``(start, end)`` index pairs.  The i-th mini-batch is
                data[mini_batch_boundaries[i][0]:mini_batch_boundaries[i][1]].

        Returns:
            ``result[i][dp_rank]`` - ObjectRef for mini-batch *i*, DP rank *dp_rank*.
        """
        all_chunk_refs: List[List[ObjectRef]] = []
        for start, end in mini_batch_boundaries:
            mini_batch = data[start:end]
            mb_size = end - start

            # Pad to make divisible by dp_size. Will only be non-zero for step-wise training.
            pad_size = (-mb_size) % dp_size
            if pad_size > 0:
                mini_batch = pad_training_input_batch(mini_batch, pad_size)

            mini_batch_size = len(mini_batch)
            assert (
                mini_batch_size % dp_size == 0
            ), f"mini_batch_size % dp_size != 0, got {mini_batch_size} and {dp_size}"
            chunk_size = mini_batch_size // dp_size
            chunks = mini_batch.chunk(chunk_size)
            all_chunk_refs.append([ray.put(chunk) for chunk in chunks])
        return all_chunk_refs

    @classmethod
    def dispatch_from_staged(
        cls,
        actor_infos: List[ActorInfo],
        method: str,
        chunk_refs: List[ObjectRef],
        **kwargs,
    ) -> List[ObjectRef]:
        """
        Dispatch pre-staged per-DP chunks to workers.

        Each worker receives only its own chunk (already in the object
        store), avoiding unnecessary deserialization overhead.

        Args:
            actor_infos: List of actor info objects
            method: Name of method to call on workers (receives a single data chunk)
            chunk_refs: Pre-staged ObjectRefs, one per DP rank (from ``stage_chunks``)
            **kwargs: Additional keyword arguments to pass to the method

        Returns:
            List of ObjectRefs for worker results
        """
        assert len(actor_infos) > 0, "actor_infos must be a non-empty list"
        object_refs = []
        for actor_info in actor_infos:
            chunk_ref = chunk_refs[actor_info.rank.dp]
            object_refs.append(getattr(actor_info.handle, method).remote(chunk_ref, **kwargs))
        return object_refs

    @classmethod
    def validate_dispatch_args(cls, *args, **kwargs) -> Tuple[Tuple, Dict[str, Any]]:
        # Extract data from either positional arg or kwarg
        if args:
            data = args[0]
            remaining_kwargs = kwargs
        elif "data" in kwargs:
            data = kwargs.pop("data")
            remaining_kwargs = kwargs
        else:
            raise ValueError("MeshDispatch requires 'data' as first positional argument or keyword argument")

        if not isinstance(data, TrainingInputBatch):
            raise ValueError(f"For MeshDispatch, `data` entry should be a `TrainingInputBatch`, got {type(data)}")
        # Pass through data as positional arg, and any other kwargs (e.g., loss_fn, loss_fn_config)
        return (data,), remaining_kwargs


class PassThroughDispatch(Dispatch):
    """PassThrough dispatch type to dispatch data to a group of actors without any sharding.

    This is useful for cases where we want to run the same method on all the actors.
    Supports methods with any number of arguments.
    """

    @classmethod
    def dispatch(cls, actor_infos: List[ActorInfo], method: str, *args, **kwargs) -> List[ObjectRef]:
        return [getattr(actor_info.handle, method).remote(*args, **kwargs) for actor_info in actor_infos]

    @classmethod
    def validate_dispatch_args(cls, *args, **kwargs) -> Tuple[Tuple, Dict[str, Any]]:
        # no validation needed just pass everything
        return args, kwargs


class DispatchRegistry:
    _registry: Dict[str, Type[Dispatch]] = {"mesh": MeshDispatch, "pass_through": PassThroughDispatch}

    @classmethod
    def register(cls, name: str, dispatch_class: Type[Dispatch]) -> None:
        """Register a new dispatch type."""
        assert issubclass(dispatch_class, Dispatch)
        cls._registry[name] = dispatch_class

    @classmethod
    def get(cls, name: str) -> Type[Dispatch]:
        """Get a registered dispatch type."""
        if name not in cls._registry:
            raise KeyError(f"Dispatch type '{name}' not registered")
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> Dict[str, Type[Dispatch]]:
        """List all registered dispatch types."""
        return cls._registry


def register_dispatch_type(name: str, dispatch_class: Type) -> None:
    DispatchRegistry.register(name, dispatch_class)


def concatenate_outputs_after_mesh_dispatch(
    actor_infos: List[ActorInfo], data_batches: List[TrainingOutputBatch]
) -> TrainingOutputBatch:
    """Concatenate data batches from different ranks after mesh dispatch.

    - Data is collected only from the primary DP rank.
    - The collected chunks are concatenated in order of DP rank to reconstruct the full data.
    """
    assert len(actor_infos) == len(data_batches), "`actor_infos` and `data_batches` must have the same length"
    shards = []
    # collect in-order
    dp_rank_to_shard = {}
    for actor_info, data_batch in zip(actor_infos, data_batches):
        if actor_info.rank.is_collection_dp_rank():
            dp_rank = actor_info.rank.dp
            dp_rank_to_shard[dp_rank] = data_batch
    for i in range(actor_infos[0].rank.dp_size):
        shards.append(dp_rank_to_shard[i])
    return TrainingOutputBatch.cat(shards)
